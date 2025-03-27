"""
title: Home Assistant Integration
author: Lenaxia
author_url: https://github.com/Lenaxia
git_url: https://github.com/Lenaxia/home-assistant-integration
funding_url: - 
description: Comprehensive Home Assistant integration with complete device control
required_open_webui_version: 0.4.0
requirements: aiohttp>=3.8.0,pydantic>=2.0.0,aiofiles>=23.1.0
version: 2.0.0
license: MIT
"""

import aiohttp
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Union, Callable,
    TypeVar, Generic, Tuple, Set, AsyncGenerator
)
from pydantic import BaseModel, Field, ValidationError, field_validator
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import aiofiles
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

# Core Enums
class Domain(str, Enum):
    LIGHT = "light"
    SWITCH = "switch"
    CLIMATE = "climate"
    MEDIA_PLAYER = "media_player"
    SENSOR = "sensor"
    BINARY_SENSOR = "binary_sensor"
    AUTOMATION = "automation"
    SCENE = "scene"
    SCRIPT = "script"
    COVER = "cover"
    FAN = "fan"

class Service(str, Enum):
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    TOGGLE = "toggle"
    SET_TEMPERATURE = "set_temperature"
    SET_HVAC_MODE = "set_hvac_mode"
    SET_PRESET_MODE = "set_preset_mode"
    SET_FAN_MODE = "set_fan_mode"
    SET_HUMIDITY = "set_humidity"
    SET_BRIGHTNESS = "set_brightness"
    SET_COLOR = "set_rgb_color"
    SET_COLOR_TEMP = "set_color_temp"
    MEDIA_PLAY = "media_play"
    MEDIA_PAUSE = "media_pause"
    VOLUME_SET = "volume_set"
    VOLUME_MUTE = "volume_mute"

class EntityCategory(str, Enum):
    CONFIG = "config"
    DIAGNOSTIC = "diagnostic"
    SYSTEM = "system"
    NONE = "none"

class DeviceClass(str, Enum):
    BATTERY = "battery"
    HUMIDITY = "humidity"
    TEMPERATURE = "temperature"
    POWER = "power"
    ENERGY = "energy"
    CURRENT = "current"
    VOLTAGE = "voltage"
    PRESSURE = "pressure"
    SIGNAL_STRENGTH = "signal_strength"
    TIMESTAMP = "timestamp"

# Exception Classes
class HAError(Exception):
    """Base exception for Home Assistant errors"""
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}

class HAConnectionError(HAError):
    """Connection-related errors"""
    pass

class HAAuthenticationError(HAError):
    """Authentication-related errors"""
    pass

class HAStateError(HAError):
    """State-related errors"""
    pass

class HAServiceError(HAError):
    """Service call related errors"""
    pass

class HAWebsocketError(HAError):
    """Websocket related errors"""
    pass

class EntityState(BaseModel):
    """Entity state model"""
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    last_changed: datetime
    last_updated: datetime
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('last_changed', 'last_updated', mode='before')
    def validate_timestamps(cls, value):
        """Parse and validate datetime fields"""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.min  # Fallback to minimum datetime
        return value

class ServiceCall(BaseModel):
    """Service call model"""
    domain: str
    service: str
    entity_id: str
    data: Dict[str, Any] = Field(default_factory=dict)

class StateChange(BaseModel):
    """State change model"""
    entity_id: str
    old_state: Optional[EntityState]
    new_state: Optional[EntityState]
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('timestamp', mode='before')
    def validate_timestamp(cls, value):
        """Validate timestamp field"""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.min
        return value

class DeviceInfo(BaseModel):
    """Device information model"""
    id: str
    name: Optional[str]
    model: Optional[str]
    manufacturer: Optional[str]
    sw_version: Optional[str]
    hw_version: Optional[str]
    area_id: Optional[str]

class AreaInfo(BaseModel):
    """Area information model"""
    id: str
    name: str
    picture: Optional[str]

# Cache Implementation
class HACache:
    """Cache implementation for Home Assistant data"""
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.timeout:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp"""
        self._cache[key] = (value, time.time())
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.timeout:
            del self._cache[key]
            return None
        return value

    def invalidate(self, key: str):
        """Remove specific key from cache"""
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        """Clear entire cache"""
        self._cache.clear()

class EventCallback(Generic[T]):
    """Generic event callback handler"""
    def __init__(
        self,
        callback: Callable[[T], None],
        event_filter: Optional[Callable[[T], bool]] = None
    ):
        self.callback = callback
        self.event_filter = event_filter

    async def __call__(self, event: T) -> None:
        """Execute callback if filter passes"""
        if self.event_filter is None or self.event_filter(event):
            await self.callback(event)

class HomeAssistantClient:
    """Core Home Assistant client implementation"""
    def __init__(self):
        self.config: Optional[dict] = None  # Stores configuration as dict now
        self.session: Optional[aiohttp.ClientSession] = None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_message_id: int = 1
        self._state_callbacks: Dict[str, List[EventCallback[StateChange]]] = {}
        self._event_emitter: Optional[Callable] = None
        self._shutting_down: bool = False
        self._cache: HACache = HACache()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Cleanup all resources"""
        self._shutting_down = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
        self._websocket = None

        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        self._cache = None
        self.config = None  # Clear config during cleanup

    async def initialize(
        self,
        url: str,
        api_key: str,
        verify_ssl: bool = True,
        timeout: int = 10,
        event_emitter: Optional[Callable] = None,
        cache_timeout: int = 30,
        websocket_retry_delay: int = 5
    ) -> None:
        """Initialize the Home Assistant connection"""
        try:
            self.config = {
                "ha_url": url,
                "ha_api_key": api_key,
                "verify_ssl": verify_ssl,
                "timeout": timeout,
                "cache_timeout": cache_timeout,
                "max_retries": 3,
                "status_indicators": True,
                "websocket_retry_delay": websocket_retry_delay
            }
            self._event_emitter = event_emitter
            self._cache = HACache(timeout=cache_timeout)
            
            # Create session first
            self.session = aiohttp.ClientSession()

            try:
                # Test connection with the created session
                try:
                    if not await self.test_connection():
                        raise HAConnectionError("Unable to connect to Home Assistant")
                except HAAuthenticationError as e:
                    await self.cleanup()
                    raise  # Re-raise authentication errors directly
                
                # Initialize websocket connection
                self._ws_task = asyncio.create_task(self._websocket_listener())

                await self._emit_status("Successfully connected to Home Assistant", done=True)
            except Exception as e:
                await self.cleanup()
                if isinstance(e, HAAuthenticationError):
                    raise HAAuthenticationError(f"Authentication failed: {str(e)}") from e
                raise HAError(f"Connection failed: {str(e)}") from e

        except Exception as e:
            await self.cleanup()
            if isinstance(e, HAAuthenticationError):
                raise HAAuthenticationError(f"Authentication failed: {str(e)}") from e
            raise HAError(f"Initialization failed: {str(e)}") from e

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Any:
        """Execute the actual HTTP request"""
        headers = {
            "Authorization": f"Bearer {self.config['ha_api_key']}",  # This line is correct as-is
            "Content-Type": "application/json",
        }

        url = urljoin(self.config['ha_url'], f"api/{endpoint}")
        retry_count = 0

        while retry_count < self.config.get('max_retries', 3):  # Use get with default
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    ssl=self.config['verify_ssl'],
                    timeout=self.config['timeout'],
                    **kwargs
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        raise HAAuthenticationError("Invalid authentication token")
                    else:
                        raise HAError(f"Request failed: {await response.text()}")

            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count == self.config['max_retries']:
                    raise HAError(f"Request to {endpoint} timed out after {retry_count} retries")
                await asyncio.sleep(1)
            except aiohttp.ClientError as e:
                raise HAConnectionError(f"Connection error: {str(e)}")
            except Exception as e:
                raise HAError(f"Unexpected error: {str(e)}")

    async def test_connection(self) -> bool:
        """Test connection to Home Assistant"""
        try:
            response = await self._make_request("GET", "")
            if response.get("message") == "API running.":
                return True
            if response.get("message") == "Unauthorized":
                await self.cleanup()
                raise HAAuthenticationError("Invalid authentication token")
            return False
        except HAAuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            await self.cleanup()
            raise
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            await self.cleanup()
            raise HAConnectionError(f"Connection failed: {str(e)}") from e

    async def _emit_status(
        self,
        message: str,
        done: bool = False,
        __event_emitter__=None
    ) -> None:
        """Emit status update if event emitter is available"""
        if __event_emitter__ and self.user_valves.status_indicators:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "description": f"Home Assistant: {message}",
                    "done": done
                }
            })

    async def _websocket_listener(self):
        """Maintain websocket connection and handle messages"""
        while not self._shutting_down:
            try:
                ws_url = self.config["ha_url"].replace('http', 'ws') + "api/websocket"
                async with self.session.ws_connect(ws_url) as websocket:
                    self._websocket = websocket

                    # Authenticate
                    auth_msg = {
                        "type": "auth",
                        "access_token": self.config["ha_api_key"]
                    }
                    await websocket.send_json(auth_msg)
                    auth_response = await websocket.receive_json()

                    if auth_response.get("type") != "auth_ok":
                        error_msg = "Websocket authentication failed"
                        if self._event_emitter:
                            await self._emit_error(error_msg, self._event_emitter)
                        raise HAAuthenticationError(error_msg)

                    # Subscribe to events
                    await websocket.send_json({
                        "id": self._ws_message_id,
                        "type": "subscribe_events",
                        "event_type": "state_changed"
                    })
                    self._ws_message_id += 1

                    # Handle messages
                    async for msg in websocket:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_websocket_message(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except aiohttp.ClientError as e:
                error_msg = f"Websocket connection error: {e}"
                logger.debug(error_msg)
                if self._event_emitter:
                    await self._emit_error(error_msg, self._event_emitter)
                if not self._shutting_down:
                    retry_msg = f"Websocket connection lost. Retrying in {self.config['websocket_retry_delay']} seconds..."
                    logger.debug(retry_msg)
                    if self._event_emitter:
                        await self._emit_status(retry_msg, self._event_emitter)
                    await asyncio.sleep(self.config['websocket_retry_delay'])
            except Exception as e:
                logger.error(f"Websocket error: {e}")
                if not self._shutting_down:
                    await asyncio.sleep(self.config.websocket_retry_delay)

    async def _handle_websocket_message(self, message: dict):
        """Handle incoming websocket messages"""
        try:
            if message.get("type") == "event" and message.get("event", {}).get("event_type") == "state_changed":
                event_data = message["event"]["data"]
                
                # Validate required fields
                if not all(k in event_data for k in ["entity_id", "old_state", "new_state"]):
                    logger.warning(f"Invalid state_changed event: {event_data}")
                    return

                try:
                    state_change = StateChange(
                        entity_id=event_data["entity_id"],
                        old_state=EntityState(**event_data["old_state"]) if event_data.get("old_state") else None,
                        new_state=EntityState(**event_data["new_state"]) if event_data.get("new_state") else None
                    )
                    
                    self._cache.invalidate(f"state_{state_change.entity_id}")

                    if state_change.entity_id in self._state_callbacks:
                        for callback in self._state_callbacks[state_change.entity_id]:
                            await callback(state_change)
                            
                except ValidationError as e:
                    logger.error(f"Invalid state change data: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Websocket message handling failed: {str(e)}")

    # Entity Management Methods
    async def get_entities(self, domain: Optional[str] = None) -> List[EntityState]:
        """Get all entities or filter by domain"""
        try:
            cache_key = f"entities_{domain if domain else 'all'}"
            cached = self._cache.get(cache_key)
            if cached:
                return cached

            states = await self._make_request("GET", "states")
            entities = [EntityState(**state) for state in states]

            if domain:
                entities = [e for e in entities if e.entity_id.startswith(f"{domain}.")]

            self._cache.set(cache_key, entities)
            return entities
        except Exception as e:
            raise HAError(f"Failed to get entities: {str(e)}")

    async def get_entity_state(self, entity_id: str) -> EntityState:
        """Get current state of a specific entity"""
        try:
            cache_key = f"state_{entity_id}"
            cached = self._cache.get(cache_key)
            if cached:
                return cached

            state = await self._make_request("GET", f"states/{entity_id}")
            entity_state = EntityState(**state)
            self._cache.set(cache_key, entity_state)
            return entity_state
        except Exception as e:
            raise HAError(f"Failed to get entity state: {str(e)}")

    _VALID_TRANSITIONS = {
        "on": ["off", "unavailable", "unknown"],
        "off": ["on", "unavailable", "unknown"],
        "unavailable": ["on", "off"],
        "unknown": ["on", "off"],
        "playing": ["paused", "idle"],
        "paused": ["playing", "idle"],
        "idle": ["playing", "paused"]
    }

    def _validate_state_transition(self, old_state: str, new_state: str) -> Optional[str]:
        """Validate state transitions with proper error message"""
        if old_state not in self._VALID_TRANSITIONS:
            return f"Unrecognized previous state: {old_state}"
        
        if new_state not in self._VALID_TRANSITIONS.get(old_state, []):
            allowed = ", ".join(self._VALID_TRANSITIONS[old_state])
            return f"Invalid transition from {old_state} to {new_state}. Allowed: {allowed}"
        
        return None

    def _validate_range(self, value: Any, min_val: float, max_val: float, name: str) -> Optional[str]:
        """Generic range validation"""
        if value is not None and not (min_val <= value <= max_val):
            return f"Invalid {name} {value} - must be between {min_val}-{max_val}"
        return None

    def _validate_string_choice(self, value: str, options: list, name: str) -> Optional[str]:
        """Generic string choice validation"""
        if value and value not in options:
            return f"Invalid {name} '{value}' - must be {', '.join(options)}"
        return None


    def _validate_service_params(self, service: str, params: dict) -> List[str]:
        """Validate parameters for service calls"""
        errors = []
        if service == "turn_on":
            if "rgb_color" in params and "color_temp" in params:
                errors.append("Cannot set both RGB color and color temperature")
        return errors

    def _sanitize_input(self, value: Any, max_length: int = 255) -> Any:
        """Sanitize user-provided values"""
        if isinstance(value, str):
            return value.strip()[:max_length]
        if isinstance(value, (list, tuple)):
            return type(value)(self._sanitize_input(v) for v in value)
        return value

    async def set_entity_state(
        self,
        entity_id: str,
        state: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> EntityState:
        """Set state of an entity"""
        try:
            # State validation
            valid_states = ["on", "off", "unavailable", "unknown"]
            if state not in valid_states:
                error_msg = f"Invalid state '{state}' - must be {', '.join(valid_states)}"
                if self._event_emitter:
                    await self._emit_error(error_msg, self._event_emitter)
                return error_msg

            # Get current state for transition validation
            try:
                current_state = await self.get_entity_state(entity_id)
                transition_error = self._validate_state_transition(current_state.state, state)
                if transition_error:
                    if self._event_emitter:
                        await self._emit_error(transition_error, self._event_emitter)
                    return transition_error
            except HAError:
                pass  # Entity doesn't exist yet

            data = {"state": state}
            if attributes:
                data["attributes"] = attributes

            response = await self._make_request(
                "POST",
                f"states/{entity_id}",
                json=data
            )

            # Invalidate cache
            self._cache.invalidate(f"state_{entity_id}")
            return EntityState(**response)
        except Exception as e:
            raise HAError(f"Failed to set entity state: {str(e)}")

    # Service Call Methods
    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        **service_data
    ) -> Dict[str, Any]:
        """Generic service call method"""
        try:
            data = {"entity_id": entity_id, **service_data}
            response = await self._make_request(
                "POST",
                f"services/{domain}/{service}",
                json=data
            )

            # Invalidate cache for affected entity
            self._cache.invalidate(f"state_{entity_id}")
            return response
        except Exception as e:
            raise HAServiceError(f"Service call failed: {str(e)}")

    # State Change Subscription
    async def subscribe_to_state_changes(
        self,
        entity_id: str,
        callback: Callable[[StateChange], None],
        event_filter: Optional[Callable[[StateChange], bool]] = None
    ) -> None:
        """Subscribe to state changes for a specific entity"""
        if entity_id not in self._state_callbacks:
            self._state_callbacks[entity_id] = []
        self._state_callbacks[entity_id].append(EventCallback(callback, event_filter))

    async def unsubscribe_from_state_changes(
        self,
        entity_id: str,
        callback: Callable[[StateChange], None]
    ) -> None:
        """Unsubscribe from state changes"""
        if entity_id in self._state_callbacks:
            self._state_callbacks[entity_id] = [
                cb for cb in self._state_callbacks[entity_id]
                if cb.callback != callback
            ]

# Domain-specific Controllers
class LightController:
    """Controller for light entities"""
    def __init__(self, client: HomeAssistantClient):
        self.client = client

    async def turn_on(
        self,
        entity_id: str,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb_color: Optional[Tuple[int, int, int]] = None,
        transition: Optional[float] = None
    ) -> Dict[str, Any]:
        """Turn on light with optional parameters"""
        data = {}
        if brightness is not None:
            data["brightness"] = min(max(brightness, 0), 255)
        if color_temp is not None:
            data["color_temp"] = color_temp
        if rgb_color is not None:
            if len(rgb_color) != 3 or any(not (0 <= v <= 255) for v in rgb_color):
                error_msg = f"Invalid RGB values {rgb_color} - all values must be 0-255"
                if self.client._event_emitter:
                    await self.client._emit_error(error_msg, self.client._event_emitter)
                return error_msg
            data["rgb_color"] = rgb_color
        if transition is not None:
            data["transition"] = transition

        return await self.client.call_service("light", "turn_on", entity_id, **data)

    async def turn_off(
        self,
        entity_id: str,
        transition: Optional[float] = None
    ) -> Dict[str, Any]:
        """Turn off light"""
        data = {}
        if transition is not None:
            data["transition"] = transition
        return await self.client.call_service("light", "turn_off", entity_id, **data)

class ClimateController:
    """Controller for climate entities"""
    def __init__(self, client: HomeAssistantClient):
        self.client = client

    async def set_temperature(
        self,
        entity_id: str,
        temperature: float,
        hvac_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set climate entity temperature"""
        data = {"temperature": temperature}
        if hvac_mode:
            data["hvac_mode"] = hvac_mode
        return await self.client.call_service(
            "climate",
            "set_temperature",
            entity_id,
            **data
        )

    async def set_hvac_mode(
        self,
        entity_id: str,
        hvac_mode: str
    ) -> Dict[str, Any]:
        """Set HVAC mode"""
        return await self.client.call_service(
            "climate",
            "set_hvac_mode",
            entity_id,
            hvac_mode=hvac_mode
        )

class MediaPlayerController:
    """Controller for media player entities"""
    def __init__(self, client: HomeAssistantClient):
        self.client = client

    async def play(self, entity_id: str) -> Dict[str, Any]:
        """Start media playback"""
        return await self.client.call_service(
            "media_player",
            "media_play",
            entity_id
        )

    async def pause(self, entity_id: str) -> Dict[str, Any]:
        """Pause media playback"""
        return await self.client.call_service(
            "media_player",
            "media_pause",
            entity_id
        )

    async def set_volume(
        self,
        entity_id: str,
        volume_level: float
    ) -> Dict[str, Any]:
        """Set media player volume"""
        volume = min(max(volume_level, 0.0), 1.0)
        return await self.client.call_service(
            "media_player",
            "volume_set",
            entity_id,
            volume_level=volume
        )
class ValidationRegistry:
    """Manage validation rules and custom validators"""
    def __init__(self):
        self.validators = {
            "range": self._validate_range,
            "choice": self._validate_string_choice,
            "entity": self._validate_entity_id
        }
        
    def add_validator(self, name: str, validator: Callable):
        self.validators[name] = validator
        
    def validate(self, rules: dict, values: dict) -> List[str]:
        errors = []
        for field, rule in rules.items():
            if field in values:
                validator = self.validators.get(rule["type"])
                if validator:
                    if error := validator(values[field], **rule.get("params", {})):
                        errors.append(f"{field}: {error}")
        return errors

# Core Valves Configuration
class Tools:
    """Home Assistant Integration Tool"""

    class Valves(BaseModel):
        class Config:
            underscore_attrs_are_private = False

        """Admin-configurable valves"""
        ha_url: str = Field(
            default="http://localhost:8123/",
            description="Home Assistant URL (e.g., http://localhost:8123)"
        )
        ha_api_key: str = Field(
            default="",
            description="Long-lived access token",
            json_schema_extra={"format": "password"}
        )
        verify_ssl: bool = Field(
            default=True,
            description="Verify SSL certificates"
        )
        timeout: int = Field(
            default=10,
            description="Request timeout in seconds"
        )
        cache_timeout: int = Field(
            default=30,
            description="Cache timeout in seconds"
        )
        max_retries: int = Field(
            default=3,
            description="Maximum number of retry attempts"
        )
        websocket_retry_delay: int = Field(
            default=5,
            description="Websocket retry delay in seconds"
        )
        status_indicators: bool = Field(
            default=True,
            description="Enable status update events"
        )
        verbose_debug: bool = Field(
            default=False,
            description="Enable verbose debug logging in chat window and console"
        )
        debug_log_level: str = Field(
            default="INFO",
            description="Logging level for debug output (DEBUG/INFO/WARNING/ERROR)",
            json_schema_extra={"choices": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        )

        @field_validator('ha_url')
        def validate_url(cls, v: str) -> str:
            """Ensure URL has valid format"""
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            return v.rstrip('/') + '/'  # Ensure trailing slash

        @field_validator('ha_api_key')
        def validate_api_key(cls, v: str) -> str:
            if not v:
                raise ValueError("API key is required")
            return v

    class UserValves(BaseModel):
        class Config:
            underscore_attrs_are_private = False

        """User-configurable valves"""
        status_indicators: bool = Field(
            default=True,
            description="Enable status update events"
        )
        verbose_errors: bool = Field(
            default=False,
            description="Show detailed error messages"
        )



    def validate_entity(domain: str):
        """Decorator factory for entity validation"""
        def decorator(func):
            async def wrapper(self, entity_id: str, *args, **kwargs):
                # Validate entity ID format
                if error := self._validate_entity_id(entity_id, domain):
                    await self._emit_error(error, kwargs.get('__event_emitter__'))
                    return error
                
                # Validate entity exists
                if error := await self._validate_entity_exists(entity_id):
                    await self._emit_error(error, kwargs.get('__event_emitter__'))
                    return error
                
                return await func(self, entity_id, *args, **kwargs)
            return wrapper
        return decorator
    
    _SENSITIVE_ATTRIBUTES = {'access_token', 'api_key', 'password', 'token', 'credentials', 'secret'}

    def _sanitize_input(self, value: Any, max_length: int = 255) -> Any:
        """Sanitize user-provided values"""
        if isinstance(value, str):
            return value.strip()[:max_length]
        if isinstance(value, (list, tuple)):
            return type(value)(self._sanitize_input(v) for v in value)
        return value

    def _filter_attributes(self, attributes: dict) -> dict:
        """Filter out sensitive attributes from exposure"""
        return {k: v for k, v in attributes.items() 
                if k.lower() not in self._SENSITIVE_ATTRIBUTES 
                and not isinstance(v, (bytes, bytearray))}

    def _validate_entity_id(self, entity_id: str, expected_domain: str) -> Optional[str]:
        """Validate entity ID format and domain"""
        if not entity_id:
            return "Entity ID cannot be empty"

        parts = entity_id.split(".")
        if len(parts) != 2:
            return "Invalid entity ID format"

        domain, _ = parts
        
        # Validate against Domain enum values
        valid_domains = [e.value for e in Domain]
        if domain not in valid_domains:
            return f"Invalid domain '{domain}' - must be one of: {', '.join(valid_domains)}"

        return None  # No error

    def _validate_range(self, value: Any, min_val: float, max_val: float, name: str) -> Optional[str]:
        """Generic range validation"""
        if value is not None and not (min_val <= value <= max_val):
            return f"Invalid {name} {value} - must be between {min_val}-{max_val}"
        return None

    def _validate_string_choice(self, value: str, options: list, name: str) -> Optional[str]:
        """Generic string choice validation"""
        if value and value not in options:
            return f"Invalid {name} '{value}' - must be {', '.join(options)}"
        return None

    async def _validate_entity_exists(self, entity_id: str) -> Optional[str]:
        """Verify entity exists in Home Assistant"""
        try:
            await self.client.get_entity_state(entity_id)
            return None
        except HAError as e:
            return f"Entity {entity_id} not found in Home Assistant: {str(e)}"

    _VALID_TRANSITIONS = {
        "light": {
            "on": ["off", "unavailable", "unknown"],
            "off": ["on", "unavailable", "unknown"],
            "unavailable": ["on", "off"],
            "unknown": ["on", "off"]
        },
        "climate": {
            "heat": ["off", "cool", "auto"],
            "cool": ["off", "heat", "auto"],
            "auto": ["off", "heat", "cool"],
            "off": ["heat", "cool", "auto"]
        },
        "media_player": {
            "playing": ["paused", "idle"],
            "paused": ["playing", "idle"],
            "idle": ["playing", "paused"]
        }
    }

    def _validate_state_change(self, entity_id: str, new_state: str) -> Optional[str]:
        """Validate state transitions per domain"""
        if not self.client:
            return "Client not initialized"
        
        domain = entity_id.split(".", 1)[0]
        current_state = self.client._cache.get(f"state_{entity_id}")
        
        if not current_state:
            return None  # No previous state to validate against
            
        allowed = self._VALID_TRANSITIONS.get(domain, {}).get(current_state.state, [])
        if new_state not in allowed:
            return f"Invalid transition from {current_state.state} to {new_state} for {domain}"
        return None


    def __init__(self) -> None:
        """Initialize the Tool with valves and user valves."""
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.citation = False  # Disable auto-citations
        self.client = None
        self.light = None
        self.climate = None
        self.media = None
        
        # Initial configuration logging
        self._configure_logging()
        if self.valves.verbose_debug:
            # Start async initialization tasks
            asyncio.create_task(self._async_init())

    async def _async_init(self):
        """Async initialization tasks"""
        await self._emit_debug(
            "Tool instance created",
            {
                "valves": self.valves.model_dump(),
                "user_valves": self.user_valves.model_dump(),
                "citation_enabled": self.citation
            },
            None
        )
        await self._emit_debug(
            "Core components initialized",
            {
                "client_initialized": self.client is not None,
                "light_controller": self.light is not None,
                "climate_controller": self.climate is not None,
                "media_controller": self.media is None
            },
            None
        )
        await self._emit_debug(
            "Logging configuration",
            {
                "log_level": self.valves.debug_log_level,
                "handlers": [h.__class__.__name__ for h in logger.handlers]
            },
            None
        )
        await self._emit_debug(
            "Initial valve states",
            {
                "ha_url": self.valves.ha_url,
                "verify_ssl": self.valves.verify_ssl,
                "timeout": self.valves.timeout,
                "cache_timeout": self.valves.cache_timeout
            },
            None
        )
        await self._emit_debug(
            "Event system initialization",
            {
                "state_callbacks_registered": len(self.client._state_callbacks) if self.client else 0,
                "websocket_connected": self.client._websocket is not None if self.client else False
            },
            None
        )


    @property
    def effective_config(self) -> dict:
        """Merge admin valves with user overrides"""
        return {
            'ha_url': self.valves.ha_url or "",
            'ha_api_key': self.valves.ha_api_key or "",
            'verify_ssl': self.valves.verify_ssl,
            'timeout': self.valves.timeout,
            'cache_timeout': self.valves.cache_timeout,
            'status_indicators': self.user_valves.status_indicators if self.user_valves else self.valves.status_indicators,
            'websocket_retry_delay': self.valves.websocket_retry_delay
        }

    @property
    def ha_api_key(self) -> str:
        return self.effective_config['ha_api_key']

    @property 
    def ha_url(self) -> str:
        return self.effective_config['ha_url']

    @property
    def current_timeout(self) -> int:
        return self.effective_config['timeout']

    @property
    def current_cache_timeout(self) -> int:
        return self.effective_config['cache_timeout']

    @property
    def status_indicators_enabled(self) -> bool:
        return self.effective_config['status_indicators']
        
    @property
    def effective_status_indicators(self) -> bool:
        return self.status_indicators_enabled

    def _configure_logging(self):
        """Set up logging based on valves"""
        level = getattr(logging, self.valves.debug_log_level, logging.INFO)
        logger.setLevel(level)
        
        if self.valves.verbose_debug:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

    def _apply_user_valves(self, __user__: Optional[dict] = None) -> None:
        """Apply user-specific valves if available"""
        try:
            if __user__ and "valves" in __user__:
                self.user_valves = self.UserValves.model_validate(__user__["valves"])
            # Keep existing user_valves if no user context provided
        except ValidationError as e:
            raise ValueError(f"Invalid user settings: {str(e)}") from e


    async def _emit_status(
        self,
        message: str,
        done: bool = False,
        __event_emitter__ = None,
        status: str = "in_progress"
    ) -> None:
        """Emit status update if enabled"""
        if __event_emitter__ is not None and self.effective_config['status_indicators']:
            try:
                if not isinstance(message, str):
                    message = str(message)
                    
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": f"Home Assistant: {message}",
                        "done": done,
                        "hidden": False
                    }
                })
            except Exception as e:
                logger.error(f"Failed to emit status: {str(e)}")

    async def _emit_error(self, error_message: str, __event_emitter__ = None, **context) -> None:
        """Emit error status with validation context"""
        if __event_emitter__ is not None and self.status_indicators_enabled:
            try:
                if not isinstance(error_message, str):
                    error_message = str(error_message)
                    
                error_data = {
                    "type": "status",
                    "data": {
                        "status": "error",
                        "description": f"Home Assistant error: {error_message}",
                        "done": True,
                        "hidden": not self.user_valves.verbose_errors
                    }
                }
                if context:
                    error_data["data"]["context"] = context
                await __event_emitter__(error_data)
            except Exception as e:
                logger.error(f"Failed to emit error: {str(e)}")

    async def _emit_debug(self, message: str, context: dict, __event_emitter__ = None) -> None:
        """Emit debug message to logs and chat if enabled"""
        exc_info = context.pop("exception", False)
        logger.debug(message, extra=context, exc_info=exc_info)
        if __event_emitter__ and self.valves.verbose_debug:
            try:
                if not isinstance(message, str):
                    message = str(message)
                    
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": f"🔧 Debug - {message}",
                        "metadata": self._filter_attributes(context) if context else {}
                    }
                })
            except Exception as e:
                logger.error(f"Failed to emit debug message: {str(e)}")

    async def _emit_entity_citation(self, entity_state, __event_emitter__ = None) -> None:
        """Emit entity state as a citation source"""
        if not __event_emitter__:
            return

        try:
            domain = entity_state.entity_id.split(".", 1)[0]
            last_updated = entity_state.last_updated or datetime.min
            
            await __event_emitter__({
                "type": "citation",
                "data": {
                    "document": [f"<div class='ha-entity-citation'>Entity: {entity_state.entity_id}<br>State: {entity_state.state}</div>"],
                    "metadata": [{
                        "type": "entity_state",
                        "date_accessed": datetime.now().isoformat(),
                        "source": entity_state.entity_id,
                        "html": True,
                        "entity_type": domain,
                        "state": entity_state.state,
                        "attributes": list(self._filter_attributes(entity_state.attributes).keys()),
                        "last_updated": last_updated.isoformat() if last_updated else ""
                    }],
                    "source": {
                        "type": "home_assistant",
                        "name": "Home Assistant",
                        "url": self.valves.ha_url,
                        "entity_id": entity_state.entity_id,
                        "integration": "home_assistant",
                        "domain": domain
                    }
                }
            })
        except Exception as e:
            logger.error(f"Failed to emit citation: {str(e)}")
            await self._emit_error("Failed to format entity citation", __event_emitter__)

    async def _init_client(self, __event_emitter__ = None) -> Optional[str]:
        """Initialize the Home Assistant client if not already initialized"""
        try:
            init_context = {"method": "_init_client", "stage": "start"}
            await self._emit_debug(
                "Starting client initialization process",
                init_context,
                __event_emitter__
            )
            
            config = self.effective_config
            init_context["config"] = config
            
            if not self.client:
                await self._emit_debug(
                    "Creating new client instance",
                    init_context,
                    __event_emitter__
                )
                
                if not config['ha_url']:
                    await self._emit_debug("Missing HA URL in configuration", init_context, __event_emitter__)
                    await self._emit_error("Missing Home Assistant URL configuration", __event_emitter__)
                    return "Home Assistant URL is not configured"
                
                if not config['ha_api_key']:
                    await self._emit_debug(
                        "Missing HA API key in configuration",
                        init_context,
                        __event_emitter__
                    )
                    await self._emit_error("Missing Home Assistant API key", __event_emitter__)
                    return "Home Assistant API key is not configured"

                await self._emit_debug("Instantiating HomeAssistantClient", init_context, __event_emitter__)
                self.client = HomeAssistantClient()
                
                init_context["client"] = str(self.client)
                await self._emit_debug("Initializing client connection", init_context, __event_emitter__)
                init_start = time.time()
                
                await self.client.initialize(
                    url=config['ha_url'],
                    api_key=config['ha_api_key'],
                    verify_ssl=config['verify_ssl'],
                    timeout=config['timeout'],
                    event_emitter=__event_emitter__,
                    cache_timeout=config['cache_timeout'],
                    websocket_retry_delay=config['websocket_retry_delay']
                )
                
                init_context["duration"] = time.time() - init_start
                await self._emit_debug(f"Client initialized in {init_context['duration']:.2f}s", init_context, __event_emitter__)
                
                await self._emit_debug("Creating domain controllers", init_context, __event_emitter__)
                self.light = LightController(self.client)
                self.climate = ClimateController(self.client)
                self.media = MediaPlayerController(self.client)
                
                controller_status = {
                    "light": bool(self.light),
                    "climate": bool(self.climate),
                    "media": bool(self.media)
                }
                await self._emit_debug("Domain controllers initialized", 
                                      {**init_context, **controller_status}, 
                                      __event_emitter__)

                await self._emit_status(
                    "Home Assistant client initialized successfully", 
                    done=True, 
                    __event_emitter__=__event_emitter__
                )
                
                if self.valves.verbose_debug:
                    await self._emit_debug(
                        "Client initialization complete", 
                        {
                            **init_context,
                            "session_active": bool(self.client.session),
                            "websocket_connected": bool(self.client._websocket)
                        }, 
                        __event_emitter__
                    )
            else:
                await self._emit_debug("Using existing client connection", init_context, __event_emitter__)
                
            return None
        except asyncio.TimeoutError:
            error_msg = "Connection timed out during initialization"
            await self._emit_error(error_msg, __event_emitter__)
            self.client = None
            raise HAConnectionError(error_msg)
        except HAAuthenticationError as e:
            await self._emit_error(f"Authentication failed: {str(e)}", __event_emitter__)
            self.client = None
            raise
        except HAConnectionError as e:
            await self._emit_error(f"Connection error: {str(e)}", __event_emitter__)
            self.client = None
            raise
        except Exception as e:
            await self._emit_error(f"Unexpected error during initialization: {str(e)}", __event_emitter__)
            self.client = None
            raise

    async def list_entities(
        self,
        domain: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None
    ) -> str:
        """
        List all entities or entities of a specific domain in Home Assistant.

        Args:
            domain: Optional domain to filter entities (e.g., 'light', 'switch')

        Returns:
            A formatted string containing entity information
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_debug(
                "Fetching entity list",
                {"domain": domain},
                __event_emitter__
            )
            domain_msg = f" for domain '{domain}'" if domain else ""
            await self._emit_status(f"Listing entities{domain_msg}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                error_msg = f"Initialization failed: {init_error}"
                await self._emit_error(error_msg, __event_emitter__)
                return error_msg

            entities = await self.client.get_entities(domain)

            if not entities:
                await self._emit_status(f"No entities found{domain_msg}", done=True, __event_emitter__=__event_emitter__)
                return "No entities found."

            result = []
            for entity in entities:
                try:
                    await self._emit_entity_citation(entity, __event_emitter__)
                    result.append(
                        f"Entity: {entity.entity_id}\n"
                        f"  State: {entity.state}\n"
                    )
                except Exception as e:
                    logger.warning(f"Skipping invalid entity {entity.entity_id}: {str(e)}")
                    continue

            await self._emit_status(f"Found {len(entities)} entities{domain_msg}", done=True, __event_emitter__=__event_emitter__)
            
            if __event_emitter__:
                if self.effective_config['status_indicators']:
                    await self._emit_status(
                        f"Found {len(entities)} entities{domain_msg}",
                        done=True,
                        __event_emitter__=__event_emitter__
                    )
                if self.valves.verbose_debug:
                    await self._emit_debug(
                        f"Entity list: {len(entities)} entities{domain_msg}",
                        {
                            "domain": domain,
                            "count": len(entities),
                            "cache_key": f"entities_{domain if domain else 'all'}"
                        },
                        __event_emitter__
                    )
            
            return "\n".join(result)
        except Exception as e:
            error_msg = f"Error listing entities: {str(e)}"
            await self._emit_debug(
                "Entity fetch failed",
                {
                "error": str(e),
                "domain": domain,
                "exception": True
            }, __event_emitter__)
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    @validate_entity("light")
    async def control_light(
        self,
        entity_id: str,
        action: str,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb_color: Optional[Tuple[int, int, int]] = None,
        transition: Optional[float] = None,
        __user__: Optional[dict] = None,
        __event_emitter__ = None,
        __event_call__ = None,
        __metadata__: Optional[dict] = None,
        __messages__: Optional[List[dict]] = None,
        __files__: Optional[List[str]] = None,
        __model__: Optional[str] = None
    ) -> str:
        """
        Control a light entity in Home Assistant.

        Args:
            entity_id: The entity ID of the light
            action: The action to perform ('turn_on' or 'turn_off')
            brightness: Optional brightness level (0-255)
            color_temp: Optional color temperature
            rgb_color: Optional RGB color tuple
            transition: Optional transition time in seconds

        Returns:
            A status message indicating success or failure
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status(f"Controlling light {entity_id}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return f"Initialization failed: {init_error}"

            # Validate parameters using reusable validators
            validations = [
                lambda: self._validate_string_choice(action, ["turn_on", "turn_off", "toggle"], "action"),
                lambda: self._validate_range(brightness, 0, 255, "brightness"),
                lambda: self._validate_range(color_temp, 153, 500, "color temperature"),
                lambda: self._validate_range(transition, 0, 300, "transition time")
            ]

            if rgb_color is not None:
                validations.append(
                    lambda: (
                        None if len(rgb_color) == 3 and all(0 <= v <= 255 for v in rgb_color)
                        else f"Invalid RGB values {rgb_color} - all values must be 0-255"
                    )
                )

            for validation in validations:
                if error := validation():
                    await self._emit_error(error, __event_emitter__=__event_emitter__)
                    return error

            if action == "turn_on":
                await self.light.turn_on(
                    entity_id,
                    brightness=brightness,
                    color_temp=color_temp,
                    rgb_color=rgb_color,
                    transition=transition
                )
                success_msg = f"Successfully turned on {entity_id}"
            elif action == "turn_off":
                await self.light.turn_off(entity_id, transition=transition)
                success_msg = f"Successfully turned off {entity_id}"
            else:
                error_msg = f"Invalid action: {action}. Must be 'turn_on' or 'turn_off'"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            await self._emit_status(
                success_msg, 
                done=True, 
                status="done", 
                __event_emitter__=__event_emitter__
            )
            return success_msg
        except Exception as e:
            error_msg = f"Error controlling light: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    async def control_climate(
        self,
        entity_id: str,
        temperature: Optional[float] = None,
        hvac_mode: Optional[str] = None,
        transition: Optional[float] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None
    ) -> str:
        """Control a climate entity with debug logging"""
        await self._emit_debug("Starting climate control", {
            "entity_id": entity_id,
            "temperature": temperature,
            "hvac_mode": hvac_mode,
            "transition": transition,
            "user": self._sanitize_input(__user__) if __user__ else None
        }, __event_emitter__)
        """
        Control a climate entity in Home Assistant.

        Args:
            entity_id: The entity ID of the climate device
            temperature: Optional target temperature
            hvac_mode: Optional HVAC mode to set
            transition: Optional transition time in seconds

        Returns:
            A status message indicating success or failure
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status(f"Controlling climate device {entity_id}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return f"Initialization failed: {init_error}"

            # Validate entity ID format
            if validation_error := self._validate_entity_id(entity_id, "climate"):
                await self._emit_error(validation_error, __event_emitter__=__event_emitter__)
                return validation_error

            # Validate temperature range
            if temperature is not None:
                if not (-50 <= temperature <= 100):
                    error_msg = f"Invalid temperature {temperature}°C - must be between -50°C and 100°C"
                    await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                    return error_msg
                if len(str(temperature).split('.')[-1]) > 1:
                    error_msg = f"Too precise temperature {temperature}°C - maximum 1 decimal place"
                    await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                    return error_msg

            # Validate HVAC mode
            valid_hvac_modes = ["heat", "cool", "auto", "off"]
            if hvac_mode and hvac_mode not in valid_hvac_modes:
                error_msg = f"Invalid HVAC mode '{hvac_mode}' - must be {', '.join(valid_hvac_modes)}"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            # Validate transition time
            if transition is not None and not (0 <= transition <= 300):
                error_msg = f"Invalid transition time {transition}s - must be 0-300 seconds"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            # Validate at least one parameter is provided
            if temperature is None and hvac_mode is None:
                error_msg = "No temperature or HVAC mode specified - must provide at least one"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            if temperature is not None:
                await self.climate.set_temperature(
                    entity_id,
                    temperature=temperature,
                    hvac_mode=hvac_mode
                )
                success_msg = f"Successfully set temperature for {entity_id}"
            elif hvac_mode is not None:
                await self.climate.set_hvac_mode(entity_id, hvac_mode)
                success_msg = f"Successfully set HVAC mode for {entity_id}"
            else:
                error_msg = "No temperature or HVAC mode specified"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            await self._emit_status(
                success_msg, 
                done=True, 
                status="done", 
                __event_emitter__=__event_emitter__
            )
            return success_msg
        except Exception as e:
            error_msg = f"Error controlling climate device: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    async def control_media_player(
        self,
        entity_id: str,
        action: str,
        volume_level: Optional[float] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None
    ) -> str:
        """Control a media player entity with debug logging"""
        await self._emit_debug(
            "Starting media player control",
            {
            "entity_id": entity_id,
            "action": action,
            "volume_level": volume_level,
            "user": self._sanitize_input(__user__) if __user__ else None
        }, __event_emitter__)
        """
        Control a media player entity in Home Assistant.

        Args:
            entity_id: The entity ID of the media player
            action: The action to perform ('play', 'pause', or 'set_volume')
            volume_level: Optional volume level (0.0-1.0)

        Returns:
            A status message indicating success or failure
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status(f"Controlling media player {entity_id}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return f"Initialization failed: {init_error}"

            # Validate entity ID format and domain
            if validation_error := self._validate_entity_id(entity_id, "media_player"):
                await self._emit_error(validation_error, __event_emitter__=__event_emitter__)
                return validation_error

            # Validate action parameter
            valid_actions = ["play", "pause", "set_volume", "stop", "next_track", "previous_track"]
            if action not in valid_actions:
                error_msg = f"Invalid action '{action}' - must be {', '.join(valid_actions)}"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            if action == "play":
                await self.media.play(entity_id)
                success_msg = f"Successfully started playback on {entity_id}"
            elif action == "pause":
                await self.media.pause(entity_id)
                success_msg = f"Successfully paused playback on {entity_id}"
            elif action == "set_volume":
                if volume_level is None:
                    error_msg = "Missing volume level for set_volume action"
                    await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                    return error_msg
                if not (0.0 <= volume_level <= 1.0):
                    error_msg = f"Invalid volume level {volume_level} - must be between 0.0 and 1.0"
                    await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                    return error_msg
                
                await self.media.set_volume(entity_id, volume_level)
                success_msg = f"Successfully set volume on {entity_id} to {volume_level*100:.0f}%"
            else:
                error_msg = f"Invalid action: {action} - must be play/pause/set_volume"
                await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
                return error_msg

            await self._emit_status(
                success_msg, 
                done=True, 
                status="done", 
                __event_emitter__=__event_emitter__
            )
            return success_msg
        except Exception as e:
            error_msg = f"Error controlling media player: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    async def get_entity_state(
        self,
        entity_id: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None
    ) -> str:
        """Get entity state with debug logging"""
        await self._emit_debug("Fetching entity state", {
            "entity_id": entity_id,
            "user": self._sanitize_input(__user__) if __user__ else None
        }, __event_emitter__)
        """
        Get the current state of an entity.

        Args:
            entity_id: The entity ID to query

        Returns:
            A formatted string containing the entity's state information
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status(f"Fetching state for {entity_id}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return init_error

            # Generic entity validation
            if validation_error := self._validate_entity_id(entity_id, ""):  # Empty expected domain
                await self._emit_error(validation_error, __event_emitter__)
                return validation_error

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return init_error

            state = await self.client.get_entity_state(entity_id)
            await self._emit_entity_citation(state, __event_emitter__)
            await self._emit_status(f"Successfully fetched state for {entity_id}", done=True, status="done", __event_emitter__=__event_emitter__)
            return (
                f"Entity: {state.entity_id}\n"
                f"State: {state.state}\n"
                f"Last Updated: {state.last_updated}\n"
                f"Attributes:\n{json.dumps(state.attributes, indent=2)}"
            )
        except Exception as e:
            error_msg = f"Error getting entity state: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    async def clear_cache(
        self,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None
    ) -> str:
        """Clear cache with debug logging"""
        debug_ctx = {
            "user": self._sanitize_input(__user__) if __user__ else None,
            "cache_size": len(self.client._cache._cache) if self.client else 0
        }
        await self._emit_debug("Clearing cache", debug_ctx, __event_emitter__)
        """
        Clear the Home Assistant entity cache.

        Returns:
            A message indicating the cache was cleared
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status("Starting cache clearance", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return init_error

            self.client._cache.clear()
            
            await self._emit_status("Cache cleared successfully", done=True, __event_emitter__=__event_emitter__)
            return "Successfully cleared Home Assistant cache"
        except Exception as e:
            error_msg = f"Error clearing cache: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

    async def monitor_entity(
        self,
        entity_id: str,
        duration: int = 60,
        __user__: Optional[dict] = None,
        __event_emitter__ = None
    ) -> str:
        """Monitor entity with debug logging"""
        debug_ctx = {
            "entity_id": entity_id,
            "duration": duration,
            "user": self._sanitize_input(__user__) if __user__ else None
        }
        await self._emit_debug("Starting entity monitoring", debug_ctx, __event_emitter__)
        """
        Monitor an entity's state changes for a specified duration.

        Args:
            entity_id: The entity ID to monitor
            duration: Duration in seconds to monitor (default: 60)

        Returns:
            A message indicating monitoring has started
        """
        try:
            self._apply_user_valves(__user__)
            await self._emit_status(f"Starting monitoring for {entity_id}", __event_emitter__=__event_emitter__)

            # Initialize client
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return init_error
            init_error = await self._init_client(__event_emitter__)
            if init_error:
                return init_error

            async def state_callback(state_change: StateChange):
                try:
                    if __event_emitter__:
                        # Validate we have required data
                        if not all([state_change.entity_id, 
                                   state_change.old_state, 
                                   state_change.new_state]):
                            logger.warning(f"Incomplete state change: {state_change}")
                            return
                        
                        await self._emit_status(
                            f"Entity {state_change.entity_id} changed from "
                            f"{state_change.old_state.state if state_change.old_state else 'unknown'} "
                            f"to {state_change.new_state.state if state_change.new_state else 'unknown'}",
                            done=False,
                            status="update",
                            __event_emitter__=__event_emitter__
                        )
                        
                        if state_change.new_state:
                            await self._emit_entity_citation(state_change.new_state, __event_emitter__)
                except Exception as e:
                    logger.error(f"State callback failed: {str(e)}")

            await self.client.subscribe_to_state_changes(entity_id, state_callback)
            
            # Emit monitoring started and final status
            await self._emit_status(f"Successfully started monitoring {entity_id}", __event_emitter__=__event_emitter__)
            await self._emit_status(
                f"Monitoring active for {duration} seconds. State changes will be reported.", 
                done=True,
                __event_emitter__=__event_emitter__
            )
            
            # Return formatted confirmation
            return f"Successfully started monitoring {entity_id} for {duration} seconds. " \
                   "State changes will be reported as they occur."
        except Exception as e:
            error_msg = f"Error setting up monitoring: {str(e)}"
            await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
            return error_msg

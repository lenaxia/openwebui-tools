"""
title: Home Assistant Integration Tool
author: projectmoon
version: 1.0.0
license: AGPL-3.0+
required_open_webui_version: 0.4.3
requirements: pydantic
"""

import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generic, TypeVar
from dataclasses import dataclass
import time
import aiohttp
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Error Handling Classes
class HAError(Exception):
    """Base error class for Home Assistant integration"""
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now()
        
    def format_error(self) -> dict:
        """Format error for API response"""
        return {
            "error": str(self),
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "type": self.__class__.__name__,
            "suggestion": self.context.get("suggestion", "Check logs for details")
        }

class HAConnectionError(HAError):
    """Connection-related errors"""
    def __init__(self, message: str, url: Optional[str] = None):
        context = {"url": url} if url else {}
        super().__init__(f"Connection error: {message}", context)

class HAAuthenticationError(HAError):
    """Authentication failures"""
    def __init__(self, message: str, auth_type: Optional[str] = None):
        context = {"auth_type": auth_type} if auth_type else {}
        super().__init__(f"Authentication error: {message}", context)

class HAStateError(HAError):
    """Invalid entity states"""
    def __init__(self, message: str, entity_id: Optional[str] = None):
        context = {"entity_id": entity_id} if entity_id else {}
        super().__init__(f"State error: {message}", context)

class HAServiceError(HAError):
    """Service call errors"""
    def __init__(self, message: str, service: Optional[str] = None):
        context = {"service": service} if service else {}
        super().__init__(f"Service error: {message}", context)

class HAWebsocketError(HAError):
    """WebSocket connection errors"""
    def __init__(self, message: str, ws_state: Optional[str] = None):
        context = {"ws_state": ws_state} if ws_state else {}
        super().__init__(f"WebSocket error: {message}", context)

class HAEntityNotFoundError(HAError):
    """Entity not found errors"""
    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity {entity_id} not found",
            {"entity_id": entity_id, "suggestion": "Check entity exists and is online"}
        )

class HAInvalidConfigError(HAError):
    """Configuration validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        context = {"field": field} if field else {}
        super().__init__(f"Configuration error: {message}", context)

class HATimeoutError(HAError):
    """Timeout errors"""
    def __init__(self, message: str, timeout: Optional[int] = None):
        context = {"timeout": timeout} if timeout else {}
        super().__init__(f"Timeout error: {message}", context)

class HACacheError(HAError):
    """Cache-related errors"""
    def __init__(self, message: str, cache_key: Optional[str] = None):
        context = {"cache_key": cache_key} if cache_key else {}
        super().__init__(f"Cache error: {message}", context)

# Core Data Models
class EntityState(BaseModel):
    """Represents the state of a Home Assistant entity"""
    entity_id: str
    state: str
    attributes: Dict[str, Any] = {}
    last_changed: datetime
    last_updated: datetime
    context: Optional[Dict[str, Any]] = None

class ServiceCall(BaseModel):
    """Represents a service call to Home Assistant"""
    domain: str
    service: str
    service_data: Dict[str, Any] = {}
    target: Optional[Dict[str, Any]] = None

class StateChange(BaseModel):
    """Represents a state change event"""
    entity_id: str
    old_state: Optional[EntityState]
    new_state: EntityState
    context: Optional[Dict[str, Any]] = None

class EventSerializer:
    """Utility class for event serialization/deserialization"""
    @staticmethod
    def serialize(event: Any) -> dict:
        """Serialize an event to a dictionary"""
        if isinstance(event, BaseModel):
            return event.model_dump()
        elif isinstance(event, dict):
            return event
        elif hasattr(event, '__dict__'):
            return vars(event)
        return {'data': str(event)}

    @staticmethod
    def deserialize(event_type: str, data: dict) -> Any:
        """Deserialize an event based on its type"""
        if event_type == 'state_change':
            return StateChange(**data)
        elif event_type == 'service_call':
            return ServiceCall(**data)
        return data

class DeviceInfo(BaseModel):
    """Contains information about a device"""
    id: str
    name: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    sw_version: Optional[str] = None
    hw_version: Optional[str] = None
    via_device_id: Optional[str] = None
    area_id: Optional[str] = None
    config_entries: List[str] = []
    connections: List[Tuple[str, str]] = []
    identifiers: List[Tuple[str, str]] = []

class AreaInfo(BaseModel):
    """Contains information about an area/zone"""
    id: str
    name: str
    picture: Optional[str] = None
    aliases: List[str] = []

# Configuration Models
class Configuration(BaseModel):
    """Home Assistant configuration settings"""
    ha_url: str
    api_key: str
    verify_ssl: bool = True
    ssl_ca_path: Optional[str] = None
    timeout: int = 10
    cache_timeout: int = 30
    event_emitter: Optional[Callable] = None

    @field_validator('ssl_ca_path')
    def validate_ssl_ca_path(cls, v: Optional[str]) -> Optional[str]:
        if v and not Path(v).exists():
            raise ValueError(f"SSL CA bundle not found at {v}")
        return v

    @field_validator('ha_url')
    def validate_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v.rstrip('/')

    @field_validator('api_key')
    def validate_api_key(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError('API key must be at least 10 characters')
        return v

    @field_validator('timeout', 'cache_timeout')
    def validate_timeouts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Timeout values must be greater than 0')
        return v

class Valves(BaseModel):
    """Tool configuration settings"""
    ha_url: str = Field(
        default="http://localhost:8123",
        description="Home Assistant instance URL"
    )
    api_key: str = Field(
        default="",
        description="Long-lived access token"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates"
    )
    timeout: int = Field(
        default=10,
        description="API request timeout in seconds"
    )
    cache_timeout: int = Field(
        default=30,
        description="Cache timeout in seconds"
    )
    status_indicators: bool = Field(
        default=True,
        description="Emit status update events"
    )

class UserValves(BaseModel):
    """User-specific configuration"""
    instruction_oriented_interpretation: bool = Field(
        default=True,
        description="Provide detailed instructions for model interpretation"
    )

class ConfigManager:
    """Handles loading and saving configuration"""
    def __init__(self, config_file: str = "hass_config.json", user_config_file: str = "user_hass_config.json"):
        self.config_file = config_file
        self.user_config_file = user_config_file
        self.config = None
        self.user_valves = None

    def load_config(self) -> Configuration:
        """Load configuration from file"""
        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
                self.config = Configuration(**data)
                return self.config
        except FileNotFoundError:
            # Return default config if file doesn't exist
            self.config = Configuration(
                ha_url="http://localhost:8123",
                api_key="default-invalid-key-please-change",
                verify_ssl=True,
                timeout=10,
                cache_timeout=30
            )
            return self.config
        except Exception as e:
            raise HAInvalidConfigError(f"Failed to load config: {str(e)}")

    def save_config(self, config: Configuration) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config.model_dump(), f, indent=2)
        except Exception as e:
            raise HAInvalidConfigError(f"Failed to save config: {str(e)}")

    def validate_config(self, config: Configuration) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate URL
        if not config.ha_url.startswith(('http://', 'https://')):
            issues.append("URL must start with http:// or https://")
            
        # Validate API key
        if len(config.api_key) < 10:
            issues.append("API key must be at least 10 characters")
            
        # Validate timeout values
        if config.timeout <= 0:
            issues.append("Timeout must be greater than 0")
            
        if config.cache_timeout <= 0:
            issues.append("Cache timeout must be greater than 0")
            
        # Validate SSL paths
        if config.ssl_ca_path and not Path(config.ssl_ca_path).exists():
            issues.append(f"SSL CA bundle not found at {config.ssl_ca_path}")
            
        return issues

    def load_user_valves(self) -> UserValves:
        """Load user-specific configuration"""
        try:
            with open(self.user_config_file, "r") as f:
                data = json.load(f)
                self.user_valves = UserValves(**data)
                return self.user_valves
        except FileNotFoundError:
            self.user_valves = UserValves()
            return self.user_valves
        except Exception as e:
            raise HAInvalidConfigError(f"Failed to load user config: {str(e)}")

    def save_user_valves(self, user_valves: UserValves) -> None:
        """Save user-specific configuration"""
        try:
            with open(self.user_config_file, "w") as f:
                json.dump(user_valves.model_dump(), f, indent=2)
        except Exception as e:
            raise HAInvalidConfigError(f"Failed to save user config: {str(e)}")

# Utility Classes
@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata"""
    value: Any
    timestamp: float
    hits: int = 0

class HACache:
    """Enhanced cache implementation for Home Assistant data"""
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'expirations': 0,
            'size': 0
        }
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired"""
        self._cleanup_expired()
        
        if key in self._cache:
            entry = self._cache[key]
            entry.hits += 1
            self._stats['hits'] += 1
            return entry.value
            
        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Any):
        """Set cached value with timestamp"""
        self._cleanup_expired()
        self._cache[key] = CacheEntry(
            value=value,
            timestamp=time.time()
        )
        self._stats['size'] = len(self._cache)

    def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        if key in self._cache:
            del self._cache[key]
            self._stats['invalidations'] += 1
            self._stats['size'] = len(self._cache)

    def invalidate_by_prefix(self, prefix: str):
        """Invalidate all entries with matching prefix"""
        to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in to_remove:
            self.invalidate(key)

    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'expirations': 0,
            'size': 0
        }

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return self._stats.copy()

    def get_entries(self) -> dict:
        """Get all cache entries with metadata"""
        return {
            key: {
                'value': entry.value,
                'age': time.time() - entry.timestamp,
                'hits': entry.hits
            }
            for key, entry in self._cache.items()
        }

    def _cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        if now - self._last_cleanup < self.timeout / 2:
            return
            
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.timestamp > self.timeout
        ]
        
        for key in expired:
            del self._cache[key]
            self._stats['expirations'] += 1
            
        self._stats['size'] = len(self._cache)
        self._last_cleanup = now

T = TypeVar('T')

class EventSystem:
    """Core event management system"""
    def __init__(self):
        self._callbacks: Dict[str, List[EventCallback]] = {}
        self._rate_limits: Dict[str, float] = {}
        self._event_history: Dict[str, List[dict]] = {}
        self._stats = {
            'events_processed': 0,
            'events_filtered': 0,
            'events_rate_limited': 0,
            'events_dropped': 0,
            'events_queued': 0
        }
        self._queue = asyncio.Queue(maxsize=1000)
        self._processing_task = None

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[Any], None],
        event_filter: Optional[Callable[[Any], bool]] = None,
        rate_limit: Optional[float] = None
    ) -> None:
        """Register a new event callback"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        
        callback_wrapper = EventCallback(callback, event_filter)
        self._callbacks[event_type].append(callback_wrapper)
        
        if rate_limit is not None:
            self._rate_limits[event_type] = rate_limit

    async def start(self):
        """Start event processing"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop event processing"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

    async def emit_event(self, event_type: str, event_data: Any) -> None:
        """Emit an event to all registered callbacks"""
        try:
            # Serialize and timestamp the event
            serialized = EventSerializer.serialize(event_data)
            event = {
                'type': event_type,
                'data': serialized,
                'timestamp': time.time()
            }
            
            # Store in history (keep last 100 events per type)
            if event_type not in self._event_history:
                self._event_history[event_type] = []
            self._event_history[event_type].append(event)
            if len(self._event_history[event_type]) > 100:
                self._event_history[event_type].pop(0)
                
            # Queue the event for processing
            try:
                self._queue.put_nowait(event)
                self._stats['events_queued'] += 1
            except asyncio.QueueFull:
                self._stats['events_dropped'] += 1
                logger.warning(f"Event queue full, dropping event: {event_type}")
                
        except Exception as e:
            logger.error(f"Error emitting event: {str(e)}")

    async def _process_events(self):
        """Background task to process events from queue"""
        while True:
            try:
                event = await self._queue.get()
                self._stats['events_processed'] += 1
                
                event_type = event['type']
                event_data = event['data']
                
                # Check rate limiting
                if event_type in self._rate_limits:
                    last_emit = self._rate_limits.get(f"{event_type}_last_emit", 0)
                    now = time.time()
                    if now - last_emit < self._rate_limits[event_type]:
                        self._stats['events_rate_limited'] += 1
                        continue
                    self._rate_limits[f"{event_type}_last_emit"] = now
                
                if event_type not in self._callbacks:
                    continue
                    
                for callback in self._callbacks[event_type]:
                    try:
                        if callback.event_filter is None or callback.event_filter(event_data):
                            await callback(event_data)
                    except Exception as e:
                        logger.error(f"Error in event callback: {str(e)}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")

    def get_stats(self) -> dict:
        """Get event system statistics"""
        return {
            **self._stats,
            'active_callbacks': sum(len(cbs) for cbs in self._callbacks.values()),
            'registered_event_types': list(self._callbacks.keys()),
            'queue_size': self._queue.qsize(),
            'history_sizes': {k: len(v) for k, v in self._event_history.items()}
        }

    def get_event_history(self, event_type: str, limit: int = 10) -> List[dict]:
        """Get recent events of a specific type"""
        return self._event_history.get(event_type, [])[-limit:]

    def clear_event_history(self, event_type: Optional[str] = None) -> None:
        """Clear event history"""
        if event_type is None:
            self._event_history.clear()
        elif event_type in self._event_history:
            del self._event_history[event_type]

    def clear_callbacks(self, event_type: Optional[str] = None) -> None:
        """Clear registered callbacks"""
        if event_type is None:
            self._callbacks.clear()
        elif event_type in self._callbacks:
            del self._callbacks[event_type]

class HomeAssistantClient:
    """Core Home Assistant client implementation"""
    def __init__(self):
        self._session = None
        self._websocket = None
        self._config = None
        self._cache = HACache()
        self._event_system = EventSystem()
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._pending_requests = {}
        self._request_counter = 0
        self._subscriptions = {}
        self._event_listener_task = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def initialize(
        self,
        url: str,
        api_key: str,
        verify_ssl: bool = True,
        timeout: int = 10,
        event_emitter: Optional[Callable] = None,
        cache_timeout: int = 30
    ) -> None:
        """Initialize the client with connection parameters"""
        self._config = Configuration(
            ha_url=url,
            api_key=api_key,
            verify_ssl=verify_ssl,
            timeout=timeout,
            cache_timeout=cache_timeout,
            event_emitter=event_emitter
        )
        self._cache = HACache(timeout=cache_timeout)
        await self._connect()

    async def _connect(self) -> None:
        """Establish connection to Home Assistant"""
        async with self._connection_lock:
            if self._connected:
                return

            try:
                self._session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=self._config.timeout)
                )
                
                # Test connection
                if not await self.test_connection():
                    raise HAConnectionError("Failed to connect to Home Assistant")

                # Start WebSocket connection
                self._websocket = await self._session.ws_connect(
                    f"{self._config.ha_url}/api/websocket"
                )
                
                # Authenticate WebSocket
                auth_response = await self._websocket.receive_json()
                if auth_response.get("type") != "auth_required":
                    raise HAAuthenticationError("WebSocket authentication required")

                await self._websocket.send_json({
                    "type": "auth",
                    "access_token": self._config.api_key
                })

                auth_result = await self._websocket.receive_json()
                if auth_result.get("type") != "auth_ok":
                    raise HAAuthenticationError("WebSocket authentication failed")

                # Start event listener
                self._event_listener_task = asyncio.create_task(self._websocket_listener())
                self._connected = True

            except Exception as e:
                await self.cleanup()
                raise HAConnectionError(f"Connection failed: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources"""
        async with self._connection_lock:
            if self._event_listener_task:
                self._event_listener_task.cancel()
                try:
                    await self._event_listener_task
                except asyncio.CancelledError:
                    pass
                self._event_listener_task = None

            if self._websocket:
                await self._websocket.close()
                self._websocket = None

            if self._session:
                await self._session.close()
                self._session = None

            self._connected = False

    async def test_connection(self) -> bool:
        """Test connection to Home Assistant"""
        try:
            async with self._session.get(
                f"{self._config.ha_url}/api/",
                ssl=self._config.verify_ssl
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> dict:
        """Make an API request to Home Assistant"""
        if not self._connected:
            raise HAConnectionError("Not connected to Home Assistant")

        cache_key = f"{method}:{endpoint}:{json.dumps(kwargs)}"
        if method == "GET" and (cached := self._cache.get(cache_key)):
            return cached

        try:
            async with self._session.request(
                method,
                f"{self._config.ha_url}/api/{endpoint}",
                ssl=self._config.verify_ssl,
                **kwargs
            ) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    raise HAError(
                        f"API request failed: {error_data.get('message')}",
                        {"status": response.status, "endpoint": endpoint}
                    )

                data = await response.json()
                if method == "GET":
                    self._cache.set(cache_key, data)
                return data
        except Exception as e:
            raise HAError(f"Request failed: {str(e)}")

    async def _websocket_listener(self):
        """Listen for WebSocket messages"""
        while self._connected:
            try:
                message = await self._websocket.receive_json()
                await self._handle_websocket_message(message)
            except Exception as e:
                if self._connected:
                    logger.error(f"WebSocket error: {str(e)}")
                    await self.cleanup()
                    break

    async def _handle_websocket_message(self, message: dict):
        """Handle incoming WebSocket messages"""
        if message.get("type") == "event":
            event = EventSerializer.deserialize(
                message["event"]["event_type"],
                message["event"]["data"]
            )
            await self._event_system.emit_event(
                message["event"]["event_type"],
                event
            )

class EventCallback(Generic[T]):
    """Enhanced event callback handler with serialization"""
    def __init__(
        self,
        callback: Callable[[T], None],
        event_filter: Optional[Callable[[T], bool]] = None,
        serialize: bool = False
    ):
        self.callback = callback
        self.event_filter = event_filter
        self.serialize = serialize
        self._last_event: Optional[T] = None

    async def __call__(self, event: T) -> None:
        if self.event_filter is None or self.event_filter(event):
            if self.serialize:
                event = self._serialize_event(event)
            self._last_event = event
            await self.callback(event)

    def _serialize_event(self, event: T) -> dict:
        """Serialize event data for storage or transmission"""
        if isinstance(event, BaseModel):
            return event.model_dump()
        elif isinstance(event, dict):
            return event
        elif hasattr(event, '__dict__'):
            return vars(event)
        return str(event)

    def get_last_event(self) -> Optional[dict]:
        """Get last processed event in serialized form"""
        if self._last_event is None:
            return None
        return self._serialize_event(self._last_event)

# Domain Enums
def rate_limited(events_per_second: float):
    """Decorator to rate limit event processing"""
    def decorator(func):
        last_called = 0.0
        
        async def wrapper(*args, **kwargs):
            nonlocal last_called
            elapsed = time.time() - last_called
            wait_time = (1.0 / events_per_second) - elapsed
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
            last_called = time.time()
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

class Domain(str, Enum):
    """Home Assistant domains"""
    LIGHT = "light"
    CLIMATE = "climate"
    MEDIA_PLAYER = "media_player"
    AUTOMATION = "automation"
    SCRIPT = "script"
    SCENE = "scene"
    SENSOR = "sensor"
    BINARY_SENSOR = "binary_sensor"
    SWITCH = "switch"
    COVER = "cover"
    FAN = "fan"
    VACUUM = "vacuum"
    CAMERA = "camera"
    LOCK = "lock"
    DEVICE_TRACKER = "device_tracker"
    PERSON = "person"
    ZONE = "zone"
    WEATHER = "weather"
    CALENDAR = "calendar"
    ALARM_CONTROL_PANEL = "alarm_control_panel"
    REMOTE = "remote"
    NOTIFY = "notify"
    IMAGE_PROCESSING = "image_processing"
    INPUT_BOOLEAN = "input_boolean"
    INPUT_NUMBER = "input_number"
    INPUT_SELECT = "input_select"
    INPUT_TEXT = "input_text"
    INPUT_DATETIME = "input_datetime"
    INPUT_BUTTON = "input_button"
    INPUT_SENSOR = "input_sensor"
    INPUT_ENTITY = "input_entity"
    INPUT_ACTION = "input_action"
    INPUT_SCRIPT = "input_script"
    INPUT_SCENE = "input_scene"
    INPUT_AUTOMATION = "input_automation"
    INPUT_ZONE = "input_zone"
    INPUT_WEATHER = "input_weather"
    INPUT_CALENDAR = "input_calendar"
    INPUT_ALARM_CONTROL_PANEL = "input_alarm_control_panel"
    INPUT_REMOTE = "input_remote"
    INPUT_NOTIFY = "input_notify"
    INPUT_IMAGE_PROCESSING = "input_image_processing"

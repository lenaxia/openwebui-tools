"""
test_home_assistant.py

Test suite for Home Assistant Integration Tool
Requirements:
    - pytest>=7.4.0
    - pytest-asyncio>=0.21.0
    - pytest-aiohttp>=1.0.4
    - pytest-mock>=3.11.1
    - aioresponses>=0.7.4
    - coverage>=7.3.0

Run tests with:
    pytest test_home_assistant.py -v --cov=home_assistant --cov-report=term-missing
"""

import pytest
import aiohttp
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, call, ANY
from aioresponses import aioresponses
from typing import Dict, Any

# Import your Home Assistant implementation
from pydantic import ValidationError
from tool import (
    Tools, HomeAssistantClient, HAError, HAConnectionError,
    HAAuthenticationError, EntityState, StateChange
)

# Test Constants
TEST_HA_URL = "https://ha.domain.tld/"
TEST_HA_TOKEN = "test_token"
TEST_ENTITY_ID = "light.test_light"
TEST_CLIMATE_ID = "climate.test_climate"
TEST_MEDIA_ID = "media_player.test_media"

# Mock Response Data
MOCK_LIGHT_STATE = {
    "entity_id": TEST_ENTITY_ID,
    "state": "on",
    "attributes": {
        "brightness": 255,
        "color_temp": 400,
        "rgb_color": [255, 0, 0]
    },
    "last_changed": "2025-03-24T10:00:00+00:00",
    "last_updated": "2025-03-24T10:00:00+00:00",
    "context": {"id": "test_context"}
}


import pytest_asyncio

import pytest_asyncio  

@pytest_asyncio.fixture
async def ha_client():
    """Create a Home Assistant client instance for testing."""
    return HomeAssistantClient()

@pytest.fixture
def mock_aioresponse():
    """Create a mock aiohttp response."""
    with aioresponses() as m:
        yield m

@pytest_asyncio.fixture
async def ha_tool(mock_aioresponse):
    """Create a Home Assistant tool instance for testing"""
    tool = Tools()
    tool.valves.ha_url = TEST_HA_URL 
    tool.valves.ha_api_key = TEST_HA_TOKEN
    
    # Mock the initial connection test
    mock_aioresponse.get(
        f"{TEST_HA_URL}api/",
        status=200,
        payload={"message": "API running."}
    )

    await tool._init_client(None)
    yield tool
    await tool.client.cleanup()

class TestUserValves:
    """Test suite for User Valves functionality"""
    
    @pytest.mark.asyncio
    async def test_user_valves_overrides(self):
        """Test user valves override defaults"""
        tool = Tools()
        tool.user_valves = tool.UserValves(
            status_indicators=False
        )
        
        # Test override values
        assert tool.effective_config['status_indicators'] is False

    @pytest.mark.asyncio
    async def test_partial_user_valves_overrides(self):
        """Test partial user valves overrides"""
        tool = Tools()
        tool.user_valves = tool.UserValves(
            status_indicators=True
        )
        
        # Test mixed values
        assert tool.current_timeout == tool.valves.timeout  # admin default
        assert tool.current_cache_timeout == tool.valves.cache_timeout  # admin default
        assert tool.status_indicators_enabled is True

    @pytest.mark.asyncio
    async def test_user_valves_application_process(self):
        """Test the full valves application process with user context"""
        tool = Tools()
        user_context = {
            "valves": tool.UserValves(
                verbose_errors=True,
                status_indicators=False
            )
        }
        
        # Apply user settings through the official method
        tool._apply_user_valves(user_context)
        
        assert tool.user_valves.status_indicators is False
        assert tool.user_valves.verbose_errors is True

    @pytest.mark.asyncio
    async def test_missing_user_context(self):
        """Test behavior when no user context is provided"""
        tool = Tools()
        tool._apply_user_valves(None)
        
        # Should maintain default values
        assert tool.current_timeout == tool.valves.timeout
        assert tool.current_cache_timeout == tool.valves.cache_timeout
        assert tool.status_indicators_enabled == tool.valves.status_indicators

    @pytest.mark.asyncio
    async def test_invalid_user_valves(self):
        """Test handling of invalid user valve types"""
        tool = Tools()
        user_context = {
            "valves": {
                "timeout": "invalid_string",  # Should be int
                "cache_timeout": True,       # Should be int
                "status_indicators": "maybe"  # Should be bool
            }
        }
        
        # Should handle type conversion errors gracefully
        with pytest.raises(ValueError):
            tool._apply_user_valves(user_context)

    @pytest.mark.asyncio
    async def test_default_valves_without_user(self):
        """Test default valves are used when no user provided"""
        tool = Tools()
        tool.user_valves = None
        
        # Test default values
        assert tool.current_timeout == tool.valves.timeout
        assert tool.current_cache_timeout == tool.valves.cache_timeout
        assert tool.status_indicators_enabled == tool.valves.status_indicators

class TestClientAuthentication:
    """Test suite for HomeAssistantClient class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_aioresponse):
        """Test client initialization."""
        # Mock the API endpoint
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."}
        )

        # Create new client instance
        client = HomeAssistantClient()
        
        # Initialize with test values
        await client.initialize(
            url=TEST_HA_URL,
            api_key=TEST_HA_TOKEN,
            verify_ssl=False  # Disable SSL verification for testing
        )
        
        try:
            # Verify initialization
            assert client.config is not None
            assert client.session is not None
            assert client.config['ha_url'] == TEST_HA_URL
            assert client.config['ha_api_key'] == TEST_HA_TOKEN
        finally:
            # Cleanup
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_authentication_error(self, ha_client, mock_aioresponse):
        """Test authentication failure handling."""
        # Mock the API endpoint to return unauthorized
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "Unauthorized"}
        )

        # Attempt initialization with invalid token
        with pytest.raises(HAAuthenticationError):
            await ha_client.initialize(
                url=TEST_HA_URL,
                api_key="invalid_token"
            )

        # Verify cleanup occurred
        assert ha_client.session is None
        assert ha_client.config is None
        assert ha_client._cache is None

    @pytest.mark.asyncio
    async def test_get_entity_state(self, ha_client, mock_aioresponse):
        """Test retrieving entity state from Home Assistant."""
        # Mock the API responses
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."}
        )
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}",
            status=200,
            payload=MOCK_LIGHT_STATE
        )

        # Initialize client first
        await ha_client.initialize(
            url=TEST_HA_URL,
            api_key=TEST_HA_TOKEN
        )

        # Get and verify entity state
        state = await ha_client.get_entity_state(TEST_ENTITY_ID)
        assert isinstance(state, EntityState)
        assert state.entity_id == TEST_ENTITY_ID
        assert state.state == "on"
        assert state.attributes["brightness"] == 255

        # Cleanup
        await ha_client.cleanup()

    @pytest.mark.asyncio
    async def test_entity_citation_emission(self, ha_tool, mock_aioresponse):
        """Test that entity citations are emitted correctly."""
        mock_emitter = AsyncMock()
        ha_tool.user_valves = ha_tool.UserValves(status_indicators=True)  # Enable citations
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=200, payload={"message": "API running."})
        mock_aioresponse.get(f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}", status=200, payload=MOCK_LIGHT_STATE)

        await ha_tool.get_entity_state(TEST_ENTITY_ID, __event_emitter__=mock_emitter)
        
        # Find the citation event from all calls
        citation_call = next(
            call for call in mock_emitter.call_args_list 
            if call[0][0].get("type") == "citation"
        )
        document = citation_call[0][0]["data"]["document"][0]
        assert TEST_ENTITY_ID in document
        # Check citation metadata structure
        # Check citation metadata structure 
        mock_emitter.assert_any_call({
            "type": "citation",
            "data": {
                "document": [ANY],  # We'll verify content separately
                "metadata": [{
                    "type": "entity_state",
                    "date_accessed": ANY,
                    "source": TEST_ENTITY_ID,
                    "html": True,
                    "entity_type": "light",
                    "state": "on",
                    "attributes": ["brightness", "color_temp", "rgb_color"],
                    "last_updated": "2025-03-24T10:00:00+00:00"
                }],
                "source": {
                    "type": "home_assistant",
                    "name": "Home Assistant",
                    "url": TEST_HA_URL,
                    "entity_id": TEST_ENTITY_ID,
                    "integration": "home_assistant",
                    "domain": "light"
                }
            }
        })

    @pytest.mark.asyncio
    async def test_set_entity_state(self, ha_client, mock_aioresponse):
        """Test updating entity state in Home Assistant."""
        """Test updating entity state in Home Assistant."""
        # Mock the API responses
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."}
        )
        mock_aioresponse.post(
            f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}",
            status=200,
            payload=MOCK_LIGHT_STATE
        )

        # Initialize client first
        await ha_client.initialize(
            url=TEST_HA_URL,
            api_key=TEST_HA_TOKEN,
            verify_ssl=False  # Disable SSL verification for testing
        )

        # Set and verify entity state
        state = await ha_client.set_entity_state(
            TEST_ENTITY_ID,
            "on",
            {"brightness": 255}
        )
        assert isinstance(state, EntityState)
        assert state.state == "on"
        assert state.attributes["brightness"] == 255

        # Cleanup
        await ha_client.cleanup()

    @pytest.mark.asyncio
    @patch('tool.HomeAssistantClient._websocket_listener', new_callable=AsyncMock)
    async def test_status_emission(self, mock_websocket, ha_tool, mock_aioresponse):
        """Test status update emission during operations"""
        mock_emitter = AsyncMock()
        
        # Mock API responses
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=200, payload={"message": "API running."})
        mock_aioresponse.get(f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}", status=200, payload=MOCK_LIGHT_STATE)

        # Execute operation with event emitter
        ha_tool.user_valves.status_indicators = True  # Enable citations
        await ha_tool.get_entity_state(TEST_ENTITY_ID, __event_emitter__=mock_emitter)

        # Verify status emission sequence
        # Verify start and done status emissions exist in any order
        expected_calls = [
            call({
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "description": "Home Assistant: Fetching state for light.test_light",
                    "done": False,
                    "hidden": False
                }
            }),
            call({
                "type": "status",
                "data": {
                    "status": "done", 
                    "description": "Home Assistant: Successfully fetched state for light.test_light",
                    "done": True,
                    "hidden": False
                }
            })
        ]
        
        # Check that all expected calls were made, ignoring any extra calls
        mock_emitter.assert_has_awaits(expected_calls, any_order=True)
        
        # Verify citation was emitted - use assert_has_awaits instead of assert_any_await
        mock_emitter.assert_has_awaits([
            call({
                "type": "citation",
                "data": ANY
            })
        ], any_order=True)

    @pytest.mark.asyncio
    async def test_citation_emission_on_list_entities(self, ha_tool, mock_aioresponse):
        """Test that citations are emitted when listing entities"""
        mock_emitter = AsyncMock()
        ha_tool.user_valves = ha_tool.UserValves(status_indicators=True)  # Enable citations
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=200, payload={"message": "API running."})
        mock_aioresponse.get(f"{TEST_HA_URL}api/states", status=200, payload=[MOCK_LIGHT_STATE])

        await ha_tool.list_entities(__event_emitter__=mock_emitter)
        
        # Find the citation event from all calls
        citation_call = next(
            call for call in mock_emitter.call_args_list 
            if call[0][0].get("type") == "citation"
        )
        document = citation_call[0][0]["data"]["document"][0]
        assert "ha-entity-citation" in document
        mock_emitter.assert_any_call({
            "type": "citation",
            "data": {
                "document": [ANY],  # We'll verify content separately
                "metadata": [{
                    "type": "entity_state",
                    "date_accessed": ANY,
                    "source": TEST_ENTITY_ID,
                    "html": True,
                    "entity_type": "light",
                    "state": "on",
                    "attributes": ["brightness", "color_temp", "rgb_color"],
                    "last_updated": "2025-03-24T10:00:00+00:00"
                }],
                "source": {
                    "type": "home_assistant", 
                    "name": "Home Assistant",
                    "url": TEST_HA_URL,
                    "entity_id": TEST_ENTITY_ID,
                    "integration": "home_assistant",
                    "domain": "light"
                }
            }
        })

class TestLightController:
    """Test suite for LightController class."""

    @pytest.mark.asyncio
    async def test_turn_on_light(self, ha_tool, mock_aioresponse):
        """Test turning on a light through the tool interface."""
        # Mock API responses
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."},
            repeat=True
        )
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}",
            status=200,
            payload=MOCK_LIGHT_STATE,
            repeat=True
        )
        mock_aioresponse.post(
            f"{TEST_HA_URL}api/services/light/turn_on",
            status=200,
            payload={"success": True},
            repeat=True  # Allow multiple calls
        )
        # Mock websocket endpoint to prevent connection attempts
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/websocket",
            status=101  # HTTP 101 Switching Protocols is required for websockets
        )

        # Execute light control
        result = await ha_tool.control_light(
            TEST_ENTITY_ID,
            "turn_on",
            brightness=255,
            __user__={"valves": Tools.UserValves(timeout=20)},
            __event_emitter__=None
        )
        
        # Verify success message
        assert "Successfully turned on" in result
        assert TEST_ENTITY_ID in result
        
        # Debug: Print all recorded requests
        print("\nRecorded requests:")
        for (method, url), requests in mock_aioresponse.requests.items():
            print(f"- {method.upper()} {url}")
            for req in requests:
                print(f"  {req.kwargs.get('json', {})}")

        # Verify the service call was made with expected parameters
        # Verify the service call was made with expected parameters
        # Verify the service call was made with expected parameters
        # Flatten all requests and check each one's method and URL
        post_requests = [
            req
            for (method, url), req_list in mock_aioresponse.requests.items()
            for req in req_list
            if method.lower() == "post"
            and "api/services/light/turn_on" in str(url)
        ]
        assert post_requests, "Expected light turn_on service call was not made"
        
        # Verify the request payload
        last_call = post_requests[-1]
        assert last_call.kwargs.get("json") == {
            "entity_id": TEST_ENTITY_ID,
            "brightness": 255
        }

class TestClimateController:
    """Test suite for ClimateController class."""

    @pytest.mark.asyncio
    async def test_set_temperature(self, ha_tool, mock_aioresponse):
        """Test setting climate temperature through the tool interface."""
        # Mock API responses
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."}
        )
        mock_aioresponse.post(
            f"{TEST_HA_URL}api/services/climate/set_temperature",
            status=200,
            payload={"success": True}
        )

        # Execute climate control
        result = await ha_tool.control_climate(
            TEST_CLIMATE_ID,
            temperature=22.5,
            __user__={"valves": Tools.UserValves(timeout=20)},
            __event_emitter__=None
        )
        
        # Verify success message
        assert "Successfully set temperature" in result
        assert TEST_CLIMATE_ID in result

class TestMediaPlayerController:
    """Test suite for MediaPlayerController class."""

    @pytest.mark.asyncio
    async def test_media_status_updates(self, ha_tool, mock_aioresponse):
        """Test status updates during media player operations"""
        mock_emitter = AsyncMock()
        mock_aioresponse.post(f"{TEST_HA_URL}api/services/media_player/media_play", status=200)
        
        await ha_tool.control_media_player(
            TEST_MEDIA_ID, 
            "play",
            __event_emitter__=mock_emitter
        )
        
        # Verify start and done status emissions
        # Allow for extra boolean checks in mock
        assert mock_emitter.await_count >= 2
        mock_emitter.assert_has_calls([
            call({
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "description": "Home Assistant: Controlling media player media_player.test_media",
                    "done": False,
                    "hidden": False
                }
            }),
            call({
                "type": "status",
                "data": {
                    "status": "done",
                    "description": "Home Assistant: Successfully started playback on media_player.test_media",
                    "done": True,
                    "hidden": False
                }
            })
        ], any_order=True)
        # Allow for __bool__ calls in the mock

    @pytest.mark.asyncio
    async def test_play_media(self, ha_tool, mock_aioresponse):
        """Test media playback through the tool interface."""
        # Mock API responses
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/",
            status=200,
            payload={"message": "API running."}
        )
        mock_aioresponse.post(
            f"{TEST_HA_URL}api/services/media_player/media_play",
            status=200,
            payload={"success": True}
        )

        # Execute media control
        result = await ha_tool.control_media_player(
            TEST_MEDIA_ID,
            "play",
            __user__={"valves": Tools.UserValves(timeout=20)},
            __event_emitter__=None
        )
        
        # Verify success message
        assert "Successfully started playback" in result
        assert TEST_MEDIA_ID in result

class TestWebsocketConnection:
    """Test suite for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, ha_tool, mock_aioresponse):
        """Test WebSocket connection and state updates."""
        # This requires more complex testing with mock websockets
        pass  # Implement detailed WebSocket tests

class TestCaching:
    """Test suite for caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_state(self, ha_tool, mock_aioresponse):
        """Test state caching."""
        # Setup initial request
        mock_aioresponse.get(
            f"{TEST_HA_URL}api/states/{TEST_ENTITY_ID}",
            status=200,
            payload=MOCK_LIGHT_STATE
        )

        # First call should hit the API
        state1 = await ha_tool.get_entity_state(TEST_ENTITY_ID)

        # Second call should use cache
        state2 = await ha_tool.get_entity_state(TEST_ENTITY_ID)

        assert state1 == state2

class TestRequestHandling:
    """Test suite for request handling functionality."""

    @pytest.mark.asyncio
    async def test_request_handling(self, ha_tool, mock_aioresponse):
        """Test handling of requests."""
        # Implement request handling tests
        pass  # Implement detailed request handling tests

    @pytest.mark.asyncio 
    async def test_error_status_emission(self, ha_tool, mock_aioresponse):
        """Test error status emission during failed operations"""
        mock_emitter = AsyncMock()
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=401)
        ha_tool.user_valves.verbose_errors = True  # Show full error details
        
        result = await ha_tool.list_entities(__event_emitter__=mock_emitter)
        
        assert "Error listing entities" in result
        mock_emitter.assert_has_calls([
            call({
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "description": "Home Assistant: Listing entities",
                    "done": False,
                    "hidden": False
                }
            }),
            call({
                "type": "status", 
                "data": {
                    "status": "error",
                    "description": "Home Assistant error: Error listing entities: Failed to get entities: Connection error: Connection refused: GET https://ha.domain.tld/api/states",
                    "done": True,
                    "hidden": False
                }
            })
        ], any_order=True)

class TestInitializationChecks:
    """Test suite for initialization checks"""
    
    @pytest.mark.asyncio
    async def test_get_entity_state_init_failure(self, ha_tool, mock_aioresponse):
        """Test entity state fetch with failed initialization"""
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=401)
        
        result = await ha_tool.get_entity_state(
            TEST_ENTITY_ID,
            __event_emitter__=AsyncMock()
        )
        # Check for either authentication or connection error messages
        assert any(msg in result.lower() for msg in ["authentication failed", "connection error"])

    @pytest.mark.asyncio
    async def test_media_player_init_failure(self, ha_tool, mock_aioresponse):
        """Test media player control with failed initialization"""
        mock_aioresponse.get(f"{TEST_HA_URL}api/", status=401)
        
        result = await ha_tool.control_media_player(
            TEST_MEDIA_ID,
            "play",
            __event_emitter__=AsyncMock()
        )
        # Check for either authentication or connection error messages
        assert any(msg in result.lower() for msg in ["authentication failed", "connection error"])

class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.mark.parametrize("entity_id,expected", [
        ("light.living_room", None),
        ("invalid_entity", "Invalid entity ID format"),
        ("climate.bedroom", None),
        ("wrong_domain.switch", "Invalid domain 'wrong_domain' - must be one of: light, switch, climate, media_player, sensor, binary_sensor, automation, scene, script, cover, fan"),
    ])
    async def test_entity_validation(self, ha_tool, entity_id, expected):
        """Test entity ID validation logic"""
        result = ha_tool._validate_entity_id(entity_id, "light")
        assert result == expected

    @pytest.mark.parametrize("current_state,new_state,expected", [
        ("on", "off", None),
        ("on", "unavailable", None),
        ("off", "on", None),
        ("on", "invalid", "Invalid transition"),
    ])
    async def test_state_transitions(self, ha_tool, current_state, new_state, expected):
        """Test state transition validation"""
        entity_id = "light.test"
        ha_tool.client._cache.set(f"state_{entity_id}", EntityState(
            entity_id=entity_id,
            state=current_state,
            attributes={},
            last_changed=datetime.now(),
            last_updated=datetime.now()
        ))
        result = ha_tool._validate_state_change(entity_id, new_state)
        if expected:
            assert expected in result
        else:
            assert result is None

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        tool = Tools()
        tool.valves.ha_url = TEST_HA_URL
        tool.valves.ha_api_key = TEST_HA_TOKEN
        
        with aioresponses() as m:
            m.get(
                f"{TEST_HA_URL}api/",
                exception=aiohttp.ClientConnectionError("Connection failed")
            )

            with pytest.raises(HAError):
                await tool._init_client(None)
                await tool.get_entity_state("nonexistent.entity")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entity_id,action,expected", [
        ("invalid_format", "turn_on", "Invalid entity ID format"),
        ("wrong_domain.climate", "turn_on", "Invalid domain 'wrong_domain'"),
        ("light_missing_dot", "turn_on", "Invalid entity ID format"),
        ("light.valid_entity", "invalid_action", "Invalid action 'invalid_action'"),
        ("light.valid_entity", "turn_on", "Successfully turned on")
    ])
    async def test_invalid_entity(self, ha_tool, mock_aioresponse, entity_id, action, expected):
        """Test various invalid entity scenarios"""
        # Mock successful API responses for valid case
        if "valid_entity" in entity_id:
            mock_aioresponse.get(
                f"{TEST_HA_URL}api/states/{entity_id}",
                status=200,
                payload=MOCK_LIGHT_STATE
            )
            mock_aioresponse.post(
                f"{TEST_HA_URL}api/services/light/turn_on",
                status=200,
                payload={"success": True}
            )
            
        result = await ha_tool.control_light(entity_id, action)
        
        if "valid_entity" in entity_id:
            assert expected in result
        else:
            assert expected in result

def test_configuration_validation():
    """Test configuration validation."""
    tool = Tools()

    # Test URL validation
    tool.valves = tool.Valves(ha_url="http://localhost:8123")
    print(f"URL after validation: '{tool.valves.ha_url}'")
    assert tool.valves.ha_url == "http://localhost:8123/"  # Validator ensures trailing slash

    # Test empty URL - should fail during model creation
    with pytest.raises(ValidationError):
        tool.valves = tool.Valves(ha_url="")  # Create new instance with invalid URL

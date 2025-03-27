# test.py
import pytest
import asyncio
from pathlib import Path
from aioresponses import aioresponses
from .hass import (
    Configuration,
    ConfigManager,
    UserValves,
    HAInvalidConfigError,
    HACache,
    EventSystem,
    HomeAssistantClient,
    HAConnectionError,
    HAAuthenticationError,
    HAError
)

def test_cache_operations():
    """Test enhanced cache functionality"""
    cache = HACache(timeout=1)
    
    # Test basic get/set
    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value"
    
    # Test expiration
    import time
    time.sleep(1.1)
    assert cache.get("test_key") is None
    
    # Test invalidation
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.invalidate("key1")
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    
    # Test prefix invalidation
    cache.set("light.kitchen", "on")
    cache.set("light.living_room", "off")
    cache.set("climate.bedroom", "heat")
    cache.invalidate_by_prefix("light.")
    assert cache.get("light.kitchen") is None
    assert cache.get("light.living_room") is None
    assert cache.get("climate.bedroom") == "heat"
    
    # Test stats
    stats = cache.get_stats()
    assert stats['hits'] >= 3
    assert stats['misses'] >= 1
    assert stats['invalidations'] == 3
    assert stats['expirations'] == 1
    
    # Test entries view
    cache.set("test", "value")
    entries = cache.get_entries()
    assert "test" in entries
    assert entries["test"]["value"] == "value"
    assert isinstance(entries["test"]["age"], float)
    assert entries["test"]["hits"] == 0
    
    # Test clear
    cache.clear()
    assert cache.get_stats()["size"] == 0

@pytest.mark.asyncio
async def test_event_system():
    """Test event system functionality"""
    event_system = EventSystem()
    
    # Start event processing
    await event_system.start()
    
    # Test basic event handling
    events_received = []
    
    async def test_callback(event):
        events_received.append(event)
    
    event_system.register_callback('test_event', test_callback)
    await event_system.emit_event('test_event', {'data': 'test'})
    
    # Give time for event processing
    # Wait for event to be processed
    for _ in range(10):  # Try up to 10 times
        if events_received:
            break
        await asyncio.sleep(0.1)
    
    assert len(events_received) == 1, f"Expected 1 event, got {len(events_received)}"
    assert events_received[0]['data'] == 'test'
    
    # Test event filtering
    filtered_events = []
    
    async def filtered_callback(event):
        filtered_events.append(event)
    
    event_system.register_callback(
        'filtered_event',
        filtered_callback,
        event_filter=lambda e: e.get('value') > 10
    )
    
    await event_system.emit_event('filtered_event', {'value': 5})
    await event_system.emit_event('filtered_event', {'value': 15})
    
    # Wait for event to be processed
    for _ in range(10):  # Try up to 10 times
        if filtered_events:
            break
        await asyncio.sleep(0.1)
    
    assert len(filtered_events) == 1, f"Expected 1 filtered event, got {len(filtered_events)}"
    assert filtered_events[0]['value'] == 15
    
    # Test rate limiting
    rate_limited_events = []
    
    async def rate_limited_callback(event):
        rate_limited_events.append(event)
    
    event_system.register_callback(
        'rate_limited_event',
        rate_limited_callback,
        rate_limit=1.0  # 1 event per second
    )
    
    await event_system.emit_event('rate_limited_event', {'data': 'first'})
    await event_system.emit_event('rate_limited_event', {'data': 'second'})
    
    await asyncio.sleep(0.1)
    assert len(rate_limited_events) == 1
    assert rate_limited_events[0]['data'] == 'first'
    
    # Test event history
    history = event_system.get_event_history('test_event')
    assert len(history) == 1
    assert history[0]['data']['data'] == 'test'
    
    # Test stats
    stats = event_system.get_stats()
    assert stats['events_processed'] >= 4
    assert stats['events_rate_limited'] == 1
    assert stats['queue_size'] == 0
    
    # Test queue overflow
    for i in range(1001):
        await event_system.emit_event('overflow_test', {'count': i})
    
    await asyncio.sleep(0.1)
    stats = event_system.get_stats()
    assert stats['events_dropped'] > 0
    
    # Stop event processing
    await event_system.stop()

@pytest.mark.asyncio
@pytest.mark.timeout(10)  # 10 second timeout
async def test_client_connection():
    """Test Home Assistant client connection"""
    client = HomeAssistantClient()
    
    try:
        # Test successful connection
        with aioresponses() as m:
            m.get("https://ha.example.com/api/", status=200)
            m.get("https://ha.example.com/api/websocket", payload={
                "type": "auth_required"
            })
            m.post("https://ha.example.com/api/websocket", payload={
                "type": "auth_ok"
            })
            
            await client.initialize(
                url="https://ha.example.com",
                api_key="test-token",
                verify_ssl=False
            )
            
            assert client._connected is True
            
        # Test cleanup
        await client.cleanup()
        assert client._connected is False
        assert client._session is None
        assert client._websocket is None
        
    except Exception as e:
        await client.cleanup()
        raise e

@pytest.mark.asyncio
async def test_client_authentication():
    """Test client authentication scenarios"""
    client = HomeAssistantClient()
    
    # Test failed authentication
    with aioresponses() as m:
        m.get("https://ha.example.com/api/", status=200)
        m.get("https://ha.example.com/api/websocket", payload={
            "type": "auth_required"
        })
        m.post("https://ha.example.com/api/websocket", payload={
            "type": "auth_invalid"
        })
        
        with pytest.raises(HAAuthenticationError):
            await client.initialize(
                url="https://ha.example.com",
                api_key="invalid-token",
                verify_ssl=False
            )
            
    # Test connection failure
    with aioresponses() as m:
        m.get("https://ha.example.com/api/", status=401)
        
        with pytest.raises(HAConnectionError):
            await client.initialize(
                url="https://ha.example.com",
                api_key="test-token",
                verify_ssl=False
            )

@pytest.mark.asyncio
async def test_api_requests():
    """Test API request handling"""
    client = HomeAssistantClient()
    
    with aioresponses() as m:
        # Setup mock responses
        m.get("https://ha.example.com/api/", status=200)
        m.get("https://ha.example.com/api/websocket", payload={
            "type": "auth_required"
        })
        m.post("https://ha.example.com/api/websocket", payload={
            "type": "auth_ok"
        })
        
        # Test GET request with caching
        m.get("https://ha.example.com/api/test", payload={"data": "test"})
        
        await client.initialize(
            url="https://ha.example.com",
            api_key="test-token",
            verify_ssl=False
        )
        
        response = await client._make_request("GET", "test")
        assert response == {"data": "test"}
        
        # Test POST request
        m.post("https://ha.example.com/api/test", payload={"result": "success"})
        response = await client._make_request("POST", "test", json={"param": "value"})
        assert response == {"result": "success"}
        
        # Test error handling
        m.get("https://ha.example.com/api/error", status=400, payload={
            "message": "Bad request"
        })
        with pytest.raises(HAError):
            await client._make_request("GET", "error")
            
    await client.cleanup()

def test_configuration_validation():
    """Test configuration validation scenarios"""
    valid_config = {
        "ha_url": "https://ha.example.com",
        "api_key": "a-valid-token-that-is-long-enough",
        "timeout": 10,
        "cache_timeout": 30
    }
    
    # Test valid config
    config = Configuration(**valid_config)
    assert config.ha_url == "https://ha.example.com"
    
    # Test invalid URLs
    with pytest.raises(ValueError):
        Configuration(**{**valid_config, "ha_url": "invalid-url"})
        
    # Test short API key
    with pytest.raises(ValueError):
        Configuration(**{**valid_config, "api_key": "short"})
        
    # Test invalid timeouts
    with pytest.raises(ValueError):
        Configuration(**{**valid_config, "timeout": 0})
        
    with pytest.raises(ValueError):
        Configuration(**{**valid_config, "cache_timeout": -5})
        
    # Test SSL CA path validation
    with pytest.raises(ValueError):
        Configuration(**{**valid_config, "ssl_ca_path": "/invalid/path"})
        
    # Test valid SSL CA path
    valid_ssl_config = {**valid_config, "ssl_ca_path": "/etc/ssl/certs/ca-certificates.crt"}
    if Path("/etc/ssl/certs/ca-certificates.crt").exists():
        config = Configuration(**valid_ssl_config)
        assert config.ssl_ca_path == "/etc/ssl/certs/ca-certificates.crt"

def test_config_manager_persistence(tmp_path):
    """Test config loading/saving lifecycle"""
    config_file = tmp_path / "test_config.json"
    user_config_file = tmp_path / "test_user_config.json"
    
    manager = ConfigManager(
        config_file=str(config_file),
        user_config_file=str(user_config_file))
    
    # Test initial load creates defaults
    config = manager.load_config()
    assert config.ha_url == "http://localhost:8123"
    
    # Test save/load roundtrip
    new_config = Configuration(
        ha_url="https://new.example.com",
        api_key="new-valid-api-key-long-enough",
        timeout=15)
    manager.save_config(new_config)
    loaded_config = manager.load_config()
    assert loaded_config.ha_url == "https://new.example.com"
    
    # Test user valves persistence
    user_valves = UserValves(instruction_oriented_interpretation=False)
    manager.save_user_valves(user_valves)
    loaded_user_valves = manager.load_user_valves()
    assert loaded_user_valves.instruction_oriented_interpretation is False
    
    # Test invalid config handling
    with open(config_file, "w") as f:
        f.write("{invalid json")
    with pytest.raises(HAInvalidConfigError):
        manager.load_config()

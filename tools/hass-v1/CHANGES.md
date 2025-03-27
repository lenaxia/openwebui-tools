# Detailed Changes for Home Assistant Tool Implementation

## 1. Tool Class Structure and Implementation

### 1.1. Standardize Tool Method Signatures

**Current Issue:** Methods have inconsistent signatures and parameter ordering.

**Required Change:** Update all public methods to follow this pattern:

```python
async def method_name(
    self,
    required_param: type,
    optional_param: Optional[type] = None,
    __user__: Optional[dict] = None,  # Always include this
    __event_emitter__ = None  # Always include this as last parameter
) -> str:  # Always return strings
    """Method docstring"""
```

**Example Fix for `control_light` method:**
```python
# BEFORE:
async def control_light(
    self,
    entity_id: str,
    action: str,
    brightness: Optional[int] = None,
    color_temp: Optional[int] = None,
    rgb_color: Optional[Tuple[int, int, int]] = None,
    transition: Optional[float] = None,
    __event_emitter__: Optional[Callable] = None
) -> str:

# AFTER:
async def control_light(
    self,
    entity_id: str,
    action: str,
    brightness: Optional[int] = None,
    color_temp: Optional[int] = None,
    rgb_color: Optional[Tuple[int, int, int]] = None,
    transition: Optional[float] = None,
    __user__: Optional[dict] = None,
    __event_emitter__: Optional[Callable] = None,
    __event_call__: Optional[Callable] = None,
    __metadata__: Optional[dict] = None,
    __messages__: Optional[List[dict]] = None,
    __files__: Optional[List[str]] = None,
    __model__: Optional[str] = None
) -> str:
```

### 1.2. Add User Settings Support

**Current Issue:** No support for user-specific settings.

**Required Change:** Add a UserSettings class and update the Tool constructor:

```python
class UserSettings(BaseModel):
    """User-specific configuration overrides"""
    timeout: Optional[int] = Field(
        default=None,
        description="Override default timeout"
    )
    cache_timeout: Optional[int] = Field(
        default=None,
        description="Override default cache timeout"
    )
    status_indicators: Optional[bool] = Field(
        default=None,
        description="Override status indicator setting"
    )

def __init__(self):
    self.base_settings = self.BaseSettings()
    self.user_settings = None
    self.client = None
    self.light = None
    self.climate = None
    self.media = None
```

### 1.3. Add User Settings Processing

**Current Issue:** No mechanism to apply user-specific settings.

**Required Change:** Add a method to process user valves:

```python
def _apply_user_valves(self, __user__: Optional[dict] = None) -> None:
    """Apply user-specific valve settings if available"""
    if not __user__ or "valves" not in __user__:
        return

    user_valves = __user__["valves"]

    # Apply overrides if they exist
    if hasattr(user_valves, "timeout") and user_valves.timeout is not None:
        self.effective_timeout = user_valves.timeout
    else:
        self.effective_timeout = self.valves.timeout

    if hasattr(user_valves, "cache_timeout") and user_valves.cache_timeout is not None:
        self.effective_cache_timeout = user_valves.cache_timeout
    else:
        self.effective_cache_timeout = self.valves.cache_timeout

    if hasattr(user_valves, "status_indicators") and user_valves.status_indicators is not None:
        self.effective_status_indicators = user_valves.status_indicators
    else:
        self.effective_status_indicators = self.valves.status_indicators
```

## 2. Event Emitter Implementation

### 2.1. Create Helper Methods for Event Emission

**Current Issue:** Inconsistent event emitter usage across methods.

**Required Change:** Add standardized helper methods:

```python
async def _emit_status(
    self,
    message: str,
    done: bool = False,
    status: str = "in_progress",
    __event_emitter__ = None
) -> None:
    """Emit status update if event emitter is available"""
    if __event_emitter__ and self.valves.status_indicators:
        await __event_emitter__({
            "type": "status",
            "data": {
                "status": status,
                "description": f"Home Assistant: {message}",
                "done": done
            }
        })

async def _emit_error(self, error_message: str, __event_emitter__ = None) -> None:
    """Emit error status if event emitter is available"""
    if __event_emitter__ and self.valves.status_indicators:
        await __event_emitter__({
            "type": "status",
            "data": {
                "status": "error",
                "description": f"Home Assistant error: {error_message}",
                "done": True
            }
        })
```

### 2.2. Add Status Updates to All Operations

**Current Issue:** Missing status updates for key operations.

**Required Change:** Add status updates at beginning and end of each method:

```python
async def get_entity_state(self, entity_id: str, __user__: Optional[dict] = None, __event_emitter__ = None) -> str:
    """Get the current state of an entity"""
    try:
        self._apply_user_valves(__user__)
        await self._emit_status(f"Fetching state for {entity_id}", __event_emitter__=__event_emitter__)

        # Method implementation...

        await self._emit_status(f"Successfully fetched state for {entity_id}", done=True, __event_emitter__=__event_emitter__)
        return result_string
    except Exception as e:
        error_msg = f"Error getting entity state: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

### 2.3. Implement Citation Events

**Current Issue:** No citation events for entity data.

**Required Change:** Add method to emit entity data as citations:

```python
async def _emit_entity_citation(self, entity_state, __event_emitter__ = None) -> None:
    """Emit entity state as a citation source"""
    if not __event_emitter__ or not self.user_valves.status_indicators:
        return

    await __event_emitter__({
        "type": "citation",
        "data": {
            "document": [entity_state.model_dump_json()],
            "metadata": [{
                "date_accessed": datetime.now().isoformat(),
                "source": entity_state.entity_id,
                "last_updated": entity_state.last_updated.isoformat()
            }],
            "source": {"name": "Home Assistant"}
        }
    })
```

## 3. Error Handling

### 3.1. Wrap All Tool Methods in Try/Except

**Current Issue:** Inconsistent error handling across methods.

**Required Change:** Ensure all methods have proper try/except blocks:

```python
async def control_climate(
    self,
    entity_id: str,
    temperature: Optional[float] = None,
    hvac_mode: Optional[str] = None,
    __user__: Optional[dict] = None,
    __event_emitter__ = None
) -> str:
    """Control a climate entity"""
    try:
        self._apply_user_valves(__user__)
        await self._emit_status(f"Controlling climate device {entity_id}", __event_emitter__=__event_emitter__)

        # Method implementation...

        await self._emit_status(f"Successfully controlled climate device {entity_id}", done=True, __event_emitter__=__event_emitter__)
        return success_message
    except Exception as e:
        error_msg = f"Error controlling climate device: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

### 3.2. Add Validation Before Operations

**Current Issue:** Missing input validation.

**Required Change:** Add validation for entity IDs and parameters:

```python
def _validate_entity_id(self, entity_id: str, expected_domain: str) -> Optional[str]:
    """Validate entity ID format and domain"""
    if not entity_id:
        return "Entity ID cannot be empty"

    parts = entity_id.split(".")
    if len(parts) != 2:
        return f"Invalid entity ID format: {entity_id}"

    domain, entity = parts
    if domain != expected_domain:
        return f"Invalid entity domain: expected {expected_domain}, got {domain}"

    return None  # No error
```

Usage in methods:

```python
# In control_light method:
validation_error = self._validate_entity_id(entity_id, "light")
if validation_error:
    return validation_error
```

## 4. Client Initialization

### 4.1. Rewrite _init_client Method

**Current Issue:** Complex lazy initialization with poor error handling.

**Required Change:** Rewrite with proper error handling and status updates:

```python
async def _init_client(self, __event_emitter__ = None) -> Optional[str]:
    """Initialize the Home Assistant client if not already initialized"""
    try:
        if self.client:
            return None  # Already initialized

        await self._emit_status("Initializing Home Assistant connection...", __event_emitter__=__event_emitter__)

        # Validate configuration
        if not self.valves.ha_url:
            return "Home Assistant URL is not configured"

        if not self.valves.ha_api_key:
            return "Home Assistant API key is not configured"

        # Create and initialize client
        self.client = HomeAssistantClient()
        await self.client.initialize(
            url=self.valves.ha_url,
            api_key=self.valves.ha_api_key,
            verify_ssl=self.valves.verify_ssl,
            timeout=self.effective_timeout,
            event_emitter=__event_emitter__,
            cache_timeout=self.effective_cache_timeout
        )

        # Initialize controllers
        self.light = LightController(self.client)
        self.climate = ClimateController(self.client)
        self.media = MediaPlayerController(self.client)

        await self._emit_status("Home Assistant connection established", done=True, __event_emitter__=__event_emitter__)
        return None
    except Exception as e:
        error_msg = f"Failed to initialize Home Assistant client: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

### 4.2. Check Initialization Result in Each Method

**Current Issue:** Methods don't check if initialization succeeded.

**Required Change:** Add initialization check to all methods:

```python
async def list_entities(self, domain: Optional[str] = None, __user__: Optional[dict] = None, __event_emitter__ = None) -> str:
    """List all entities or entities of a specific domain"""
    try:
        self._apply_user_valves(__user__)
        await self._emit_status(f"Listing entities{f' for domain {domain}' if domain else ''}", __event_emitter__=__event_emitter__)

        # Initialize client
        init_error = await self._init_client(__event_emitter__)
        if init_error:
            return init_error

        # Rest of method implementation...

    except Exception as e:
        error_msg = f"Error listing entities: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

## 5. Method Implementation Improvements

### 5.1. Improve list_entities Method

**Current Issue:** Basic implementation with minimal formatting.

**Required Change:** Enhance with better formatting and filtering:

```python
async def list_entities(self, domain: Optional[str] = None, __user__: Optional[dict] = None, __event_emitter__ = None) -> str:
    """List all entities or entities of a specific domain"""
    try:
        self._apply_user_valves(__user__)
        domain_msg = f" for domain '{domain}'" if domain else ""
        await self._emit_status(f"Listing entities{domain_msg}", __event_emitter__=__event_emitter__)

        # Initialize client
        init_error = await self._init_client(__event_emitter__)
        if init_error:
            return init_error

        entities = await self.client.get_entities(domain)

        if not entities:
            return f"No entities found{domain_msg}."

        # Group entities by domain for better organization
        domain_groups = {}
        for entity in entities:
            entity_domain = entity.entity_id.split('.')[0]
            if entity_domain not in domain_groups:
                domain_groups[entity_domain] = []
            domain_groups[entity_domain].append(entity)

            # Emit citation for each entity
            await self._emit_entity_citation(entity, __event_emitter__)

        # Format results
        result = []
        for domain_name, domain_entities in sorted(domain_groups.items()):
            result.append(f"\n## {domain_name.capitalize()} Entities ({len(domain_entities)})")
            for entity in domain_entities:
                result.append(f"- **{entity.entity_id}**: {entity.state}")

                # Add important attributes if they exist
                important_attrs = []
                for attr in ['friendly_name', 'device_class', 'unit_of_measurement']:
                    if attr in entity.attributes:
                        important_attrs.append(f"{attr}={entity.attributes[attr]}")

                if important_attrs:
                    result.append(f"  ({', '.join(important_attrs)})")

        await self._emit_status(f"Found {len(entities)} entities{domain_msg}", done=True, __event_emitter__=__event_emitter__)
        return "\n".join(result)
    except Exception as e:
        error_msg = f"Error listing entities: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

### 5.2. Improve control_light Method

**Current Issue:** Basic implementation with minimal validation.

**Required Change:** Add better validation and response formatting:

```python
async def control_light(
    self,
    entity_id: str,
    action: str,
    brightness: Optional[int] = None,
    color_temp: Optional[int] = None,
    rgb_color: Optional[Tuple[int, int, int]] = None,
    transition: Optional[float] = None,
    __user__: Optional[dict] = None,
    __event_emitter__: Optional[Callable] = None,
    __event_call__: Optional[Callable] = None,
    __metadata__: Optional[dict] = None,
    __messages__: Optional[List[dict]] = None,
    __files__: Optional[List[str]] = None,
    __model__: Optional[str] = None
) -> str:
    """Control a light entity"""
    try:
        self._apply_user_valves(__user__)
        await self._emit_status(f"Controlling light {entity_id}", __event_emitter__=__event_emitter__)

        # Validate entity ID
        validation_error = self._validate_entity_id(entity_id, "light")
        if validation_error:
            return validation_error

        # Validate action
        if action not in ["turn_on", "turn_off", "toggle"]:
            return f"Invalid action: {action}. Must be 'turn_on', 'turn_off', or 'toggle'"

        # Initialize client
        init_error = await self._init_client(__event_emitter__)
        if init_error:
            return init_error

        # Validate parameters
        if brightness is not None and (brightness < 0 or brightness > 255):
            return f"Invalid brightness: {brightness}. Must be between 0 and 255"

        # Get current state for better response
        try:
            current_state = await self.client.get_entity_state(entity_id)
            await self._emit_entity_citation(current_state, __event_emitter__)
        except Exception:
            current_state = None

        # Execute action
        if action == "turn_on":
            data = {}
            if brightness is not None:
                data["brightness"] = brightness
            if color_temp is not None:
                data["color_temp"] = color_temp
            if rgb_color is not None:
                data["rgb_color"] = rgb_color
            if transition is not None:
                data["transition"] = transition

            await self.light.turn_on(entity_id, **data)

            # Build detailed response
            response_parts = [f"Successfully turned on {entity_id}"]
            if brightness is not None:
                response_parts.append(f"brightness set to {brightness}")
            if color_temp is not None:
                response_parts.append(f"color temperature set to {color_temp}K")
            if rgb_color is not None:
                response_parts.append(f"color set to RGB{rgb_color}")

            response = " with ".join(response_parts)
        elif action == "turn_off":
            await self.light.turn_off(entity_id, transition=transition)
            response = f"Successfully turned off {entity_id}"
        else:  # toggle
            await self.client.call_service("light", "toggle", entity_id)
            new_state = "on" if current_state and current_state.state == "off" else "off"
            response = f"Successfully toggled {entity_id} to {new_state}"

        # Get updated state
        try:
            new_state = await self.client.get_entity_state(entity_id)
            await self._emit_entity_citation(new_state, __event_emitter__)
        except Exception:
            pass

        await self._emit_status(f"Successfully controlled light {entity_id}", done=True, __event_emitter__=__event_emitter__)
        return response
    except Exception as e:
        error_msg = f"Error controlling light: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

## 6. Documentation and Docstrings

### 6.1. Standardize All Docstrings

**Current Issue:** Inconsistent docstring format.

**Required Change:** Update all docstrings to follow this pattern:

```python
async def get_entity_state(self, entity_id: str, __user__: Optional[dict] = None, __event_emitter__ = None) -> str:
    """
    Get the current state of an entity in Home Assistant.

    Args:
        entity_id: The entity ID to query (e.g., 'light.living_room')
        __user__: Optional user context with valve settings
        __event_emitter__: Optional event emitter for status updates

    Returns:
        A formatted string containing the entity's state information
    """
```

### 6.2. Add Method Type Hints

**Current Issue:** Inconsistent or missing type hints.

**Required Change:** Ensure all methods have proper type hints:

```python
from typing import Optional, Dict, List, Any, Tuple, Callable, Union

# Example method with complete type hints
async def control_climate(
    self,
    entity_id: str,
    temperature: Optional[float] = None,
    hvac_mode: Optional[str] = None,
    __user__: Optional[Dict[str, Any]] = None,
    __event_emitter__: Optional[Callable] = None
) -> str:
    """Method docstring"""
```

## 7. Simplification Opportunities

### 7.1. Simplify Cache Implementation

**Current Issue:** Complex cache implementation.

**Required Change:** Replace with simpler implementation:

```python
class SimpleCache:
    """Simple time-based cache implementation"""
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._cache = {}  # key -> (value, timestamp)

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired"""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.timeout:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp"""
        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """Remove specific key"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear entire cache"""
        self._cache.clear()
```

### 7.2. Simplify Error Classes

**Current Issue:** Too many specific error classes.

**Required Change:** Consolidate to fewer error types:

```python
class HAError(Exception):
    """Base exception for Home Assistant errors"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)

class HAConnectionError(HAError):
    """Connection-related errors"""
    pass

class HAAuthenticationError(HAError):
    """Authentication-related errors"""
    pass

class HAServiceError(HAError):
    """Service call and entity state errors"""
    pass
```

## 8. Additional Features to Match OSM

## Citation CSS Requirements

For proper rendering of entity citations, these CSS styles should be included:

```css
.ha-entity-citation {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    margin: 1rem 0;
    padding: 1.25rem;
    background: #f8f9fa;
}

.ha-entity-id {
    color: #2c3e50;
    margin-top: 0;
    margin-bottom: 0.75rem;
}

.ha-state-info {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 1rem;
}

.ha-state-info strong {
    color: #34495e;
}

.ha-attributes-json {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 3px;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.4;
}

.ha-last-updated {
    font-style: italic;
}
```

### 8.1. Add Result Formatting Helpers

**Current Issue:** Inconsistent result formatting.

**Required Change:** Add helper methods for consistent formatting:

```python
def _format_entity_state(self, entity: EntityState) -> str:
    """Format entity state as a readable string"""
    result = [
        f"## {entity.entity_id}",
        f"**State:** {entity.state}",
        f"**Last Updated:** {entity.last_updated.isoformat()}"
    ]

    # Add important attributes first
    for key in ['friendly_name', 'device_class', 'unit_of_measurement']:
        if key in entity.attributes:
            result.append(f"**{key.replace('_', ' ').title()}:** {entity.attributes[key]}")

    # Add other attributes
    result.append("\n**Other Attributes:**")
    for key, value in entity.attributes.items():
        if key not in ['friendly_name', 'device_class', 'unit_of_measurement']:
            result.append(f"- **{key}:** {value}")

    return "\n".join(result)
```

### 8.2. Add Cache Control Methods

**Current Issue:** No way to control cache behavior.

**Required Change:** Add methods to manage cache:

```python
async def clear_cache(self, __user__: Optional[dict] = None, __event_emitter__ = None) -> str:
    """
    Clear the Home Assistant entity cache.

    Returns:
        A message indicating the cache was cleared
    """
    try:
        self._apply_user_valves(__user__)
        await self._emit_status("Clearing cache", __event_emitter__=__event_emitter__)

        # Initialize client
        init_error = await self._init_client(__event_emitter__)
        if init_error:
            return init_error

        # Clear cache
        self.client._cache.clear()

        await self._emit_status("Cache cleared successfully", done=True, __event_emitter__=__event_emitter__)
        return "Home Assistant cache cleared successfully"
    except Exception as e:
        error_msg = f"Error clearing cache: {str(e)}"
        await self._emit_error(error_msg, __event_emitter__=__event_emitter__)
        return error_msg
```

By implementing these detailed changes, the Home Assistant tool will achieve the same level of robustness and reliability as the OpenStreetMap tool, while maintaining a consistent interface pattern that works well with LLMs.

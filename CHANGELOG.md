### 2026-03-30 — Phase 3: MoE Model Optimization

**Bug Fixes**
- Fixed `dry_sequence_breakers` not being included in API requests. The field was defined in `SamplingParams` TypedDict but was missing from `_build_sampling_kwargs()`, meaning it was silently dropped on every request.
- Fixed `samplers` key missing from the `sampling_params` dictionary in `process_request`, causing the user-specified sampler chain to never be sent to the server.

**New: MoE Optimization Mode (`moe_mode`)**
- Added `moe_mode` (BOOLEAN, default `False`) toggle to `LlamaCppClientNode`.
- When enabled, automatically applies a lightweight preset optimised for Mixture-of-Experts models:
  - Disables `timings_per_token`, `n_probs`, and `post_sampling_probs` (high overhead on MoE).
  - Simplifies the sampler chain from the full default to `["top_k", "top_p", "temperature"]`.
  - `cache_prompt` remains `True` (KV-cache reuse is especially beneficial on MoE).

**New: Server Status & Model Detection API**
- Added `get_health()` to `LlamaCppAPIClient` — queries `GET /health` to check server readiness and slot availability.
- Added `get_props()` — queries `GET /props` to retrieve server/model configuration metadata.
- Added `is_moe_model()` with two-stage detection:
  1. **Structural check**: recursively finds the `expert_count` key anywhere in the props tree (catches DeepSeek, Mixtral, and future MoE variants that expose this field explicitly).
  2. **Keyword scan**: falls back to searching the serialised props string for known MoE architecture names (`mixtral`, `deepseek`, `moe`, `experts`).

**New: Timing Metadata Output**
- The `timings` object returned by llama-server (e.g. `predicted_per_second`, `prompt_ms`) is now extracted and stored in the node's `metadata` output, making inference speed visible without enabling `timings_per_token`.

**Testing (+18 tests → 139 total, 0 lint errors)**
- `TestMoEDetection` (9): `get_health`, `get_props`, `is_moe_model` structural + keyword detection, dense-model non-detection.
- `TestMoENodeFeatures` (6): `moe_mode` preset overrides for `n_probs`, `timings_per_token`, `post_sampling_probs`, sampler simplification, `timings` extraction, graceful handling when `timings` is absent.
- `TestMoEModeUI` (3): `moe_mode` present in `INPUT_TYPES`, correct type (`BOOLEAN`), correct default (`False`).
- All tests are mock-based (no live server required). `flake8` lint passes with 0 errors.

### 2026-03-29
- Implemented full support for all previously unimplemented endpoints: completion, chat_completions, embeddings, tokenize, detokenize, apply_template, infill, reranking.
  - Extended EndpointType Enum to 8 types (COMPLETION, CHAT_COMPLETION, EMBEDDING, TOKENIZE, DETOKENIZE, INFILL, RERANKING, APPLY_TEMPLATE).
  - Separated validation and content-cleaning logic:
    - Added _GENERATION_ENDPOINTS frozenset.
    - Implemented _validate_response_structure() to return endpoint-specific validation results.
    - Implemented _apply_content_cleaning() and applied it only to generation endpoints.
    - Reworked _parse_and_validate() into a 4-phase flow (HTTP status → JSON parse → structure validation → content cleaning).
  - Expanded llamacpp_client_node.py::_extract_response_text to handle endpoint-specific response formats (chat_completions, embeddings, tokenize, reranking, detokenize, apply_template, completion, infill).
  - Updated all handle_* methods to use the correct EndpointType.
  - Added 20+ unit tests and fixed mocks; test suite reports 109 passed (2 warnings).
  - Minor UI adjustments in web/llamacpp_client_extension.js.

### 2026-03-28
- Completed comprehensive test suite for core functionality.
  - Added `tests/test_image_utils.py` with 8 tests for image processing utilities.
    - `test_detect_image_format`: Validates PNG format detection
    - `test_extract_image_metadata`: Verifies metadata extraction
    - `test_extract_tensor_metadata`: Tests tensor metadata extraction (batch_size, channels)
    - `test_validate_image_data`: Confirms data URI validation
    - `test_tensor_to_base64_data_uri`: Validates tensor to base64 conversion
    - `test_build_vision_content`: Tests vision content building with text and image
    - `test_process_image_data_string`: Validates image data string processing
  - Added `tests/test_llama_client.py` with 5 tests for API client.
    - `test_make_request_success`: HTTP 200 success handling
    - `test_make_request_failure`: HTTP 500 error handling
    - `test_handle_completion`: Completion endpoint with mocked requests
    - `test_handle_chat_completions`: Chat completions endpoint with mocked requests
  - Added `tests/test_param_utils.py` with 20+ tests for parameter utilities.
    - JSON parameter parsing tests (stop_sequences, logit_bias, messages, etc.)
    - Parameter cleaning and validation tests
    - Type conversion tests (safe_convert_to_int, safe_convert_to_float)
    - Parameter mapping tests
  - Added `tests/test_node.py` with 3 tests for node functionality.
    - `test_process_request_success`: Successful request processing
    - `test_process_request_failure`: Error handling in request processing
    - `test_input_types`: Validates INPUT_TYPES structure
  - Added `tests/test_ui_logic.py` with 2 tests for UI updates.
    - `test_update_ui_success`: Successful UI update
    - `test_update_ui_failure`: UI update error handling
  - All tests use mock-based approach (unittest.mock) for isolated testing without requiring live server.
  - Tests cover parameter validation, type conversion, image processing, and API client functionality.

### 2026-03-22
- Improved system message placement logic in `chat_completions` endpoint.
  - Refactored message building logic in `llamacpp_client_node.py` to use a stack-based approach that guarantees system message is always placed at index 0.
  - Implemented priority-based message construction:
    1. System prompt (highest priority)
    2. History messages (with duplicate prevention)
    3. User message and vision content integration
    4. Assistant message
  - Added safety net fallback to insert empty system prompt if not present.
- Added comprehensive debug logging in `llamacpp_client_node.py`.
  - Added detailed logging for all input parameters in `process_request` method.
  - Added system message position verification logic with pass/fail indicators.
- Enhanced metadata extraction functionality in `utils/image_utils.py`.
  - Added `extract_metadata` parameter to `build_vision_content` function.
  - Enhanced image metadata extraction capabilities.
  - Added `batch_index` parameter to `extract_tensor_metadata` function.
- Improved type hints in `utils/llama_client.py`.
  - Organized type hint imports.
  - Added explicit type definitions.

### 2026-03-21
- Fixed `process_request` parameter order mismatch.
  - Added missing `system_message` and `user_message` parameters to match `INPUT_TYPES` definition.
  - Reordered parameters to ensure ComfyUI passes arguments in the correct order.
- Updated `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` to use `"LlamaCppClientNode"` as the key.
- Restored and completed `process_request` method for Phase 8: Multimodal Integration.
  - Updated `RETURN_TYPES` to: `("STRING", "STRING", "STRING", "INT", "JSON")`
  - Updated `RETURN_NAMES` to: `("response", "raw_response", "error", "status_code", "metadata")`
  - Re-implemented full `process_request` body with `extract_image_metadata` and `build_vision_content` functionality.
  - Removed duplicate `extract_image_metadata_from_tensor` function; now uses `extract_tensor_metadata` from `utils/image_utils.py`.
- Fixed `test_build_vision_content_with_tensor` test in `test_optimized_features.py`.
  - Corrected unpacking of tuple return value from `build_vision_content` function.
- Fixed parameter display bug during endpoint switching.
  - Completed `endpointFields` definition in `web/llamacpp_client_extension.js` with all required parameters for each endpoint.
  - `completion` endpoint: 44 parameters
  - `chat_completions` endpoint: 20 parameters
  - `infill` endpoint: 14 parameters
  - Other endpoints also have their required parameters defined
- Improved JavaScript extension robustness.
  - Added fallback for `node.masterWidgets` copy operation (when undefined).
  - Added safety check for `images` pin detection logic (check for undefined `input`).
- Added numpy array support in `build_vision_content` function.
  - Modified `utils/image_utils.py` to handle both `torch.Tensor` and `numpy.ndarray` types.
  - This allows the function to process numpy arrays in addition to torch tensors.
- Updated test tensor shape in `test_optimized_features.py`.
  - Changed `test_build_vision_content_with_tensor` to use ComfyUI standard 4D tensor shape `(1, 256, 256, 3)` instead of 3D shape.
  - This aligns with ComfyUI's IMAGE tensor format [Batch, H, W, C].

### 2026-03-20
- Fixed `image_data` widget visibility toggle issue.
  - Changed approach to dynamically remove and rebuild unnecessary widgets from `node.widgets` array to resolve issues with HTML elements not being properly hidden and conflicts with LiteGraph rendering logic.
  - This fundamentally fixed display bugs where widgets like `image_data` and `lora` were overflowing outside the node frame.
- Moved extension JavaScript file from `web/js/` to `web/` root directory to resolve loading issues in ComfyUI environment.
  - Also fixed related import paths.
- Completely reviewed and updated UI update logic during `endpoint` switching for a more robust implementation.

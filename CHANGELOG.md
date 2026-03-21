### 2026-03-21
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
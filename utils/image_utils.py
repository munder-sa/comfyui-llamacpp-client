import base64
import gc
import io
import json
import logging
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    from logger import log_debug, log_error, log_info
except ImportError:
    try:
        from .logger import log_debug, log_error, log_info
    except ImportError:
        try:
            from utils.logger import log_debug, log_error, log_info
        except ImportError:
            try:
                from utils.logger import log_debug, log_error, log_info
            except ImportError:
                from utils.logger import log_debug, log_error, log_info

# Set up logger
logger = logging.getLogger(__name__)

# Performance optimization constants
DEFAULT_JPEG_QUALITY = 85  # Balance between quality and file size
MAX_IMAGE_DIMENSION = 2048  # Maximum dimension for resizing large images
MIN_IMAGE_DIMENSION = 512  # Minimum dimension to maintain reasonable quality


def tensor_to_base64_data_uri(
    tensor_image,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_dimension: Optional[int] = None,
    clear_memory: bool = True,
) -> str:
    """Convert ComfyUI image tensor to Base64 Data URI with performance optimizations.
    
    Args:
        tensor_image: PIL Image or torch.Tensor or numpy array
        jpeg_quality: JPEG quality (1-100), higher = better quality but larger file
        max_dimension: Maximum dimension to resize to (preserves aspect ratio)
        clear_memory: Whether to explicitly clear memory after processing
    
    Returns:
        Base64 encoded data URI string or None on error
    """
    original_tensor = None
    try:
        # Store reference for memory clearing
        if isinstance(tensor_image, torch.Tensor):
            original_tensor = tensor_image
        
        # Convert tensor to numpy if needed
        if isinstance(tensor_image, torch.Tensor):
            tensor_image = tensor_image.cpu().numpy()

        # Convert to PIL Image
        i = 255.0 * tensor_image
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Resize if image is too large
        if max_dimension is None:
            max_dimension = MAX_IMAGE_DIMENSION
        
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            log_debug(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")

        # Convert RGBA or other modes to RGB to avoid JPEG save errors
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to BytesIO as JPEG with optimized quality
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)

        # Encode to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        log_debug(
            f"Converted ComfyUI IMAGE to Base64 Data URI. "
            f"Original Mode: {img.mode}, JPEG Base64 length: {len(img_str)}, "
            f"Quality: {jpeg_quality}"
        )
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        log_error(f"Error converting image tensor to Base64: {e}", e)
        return None
    finally:
        # Clear memory if requested
        if clear_memory and original_tensor is not None:
            try:
                del original_tensor
                if torch is not None:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            except Exception:
                pass


def build_vision_content(
    user_text: str,
    image_data: list,
    tensor_images=None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_dimension: Optional[int] = None,
    clear_memory: bool = True,
) -> list:
    """Build OpenAI Vision API compatible content array from text and images.
    
    Args:
        user_text: Text content for the user message
        image_data: List of image data objects from JSON parsing
        tensor_images: ComfyUI IMAGE tensor(s)
        jpeg_quality: JPEG quality for tensor image conversion
        max_dimension: Maximum dimension for resizing
        clear_memory: Whether to clear memory after processing
    
    Returns:
        List of content items for the API request
    """
    user_content_items = []

    # 1. Add Text
    if user_text and user_text.strip():
        user_content_items.append({"type": "text", "text": user_text.strip()})

    # 2. Add Images from image_data (JSON strings)
    if isinstance(image_data, list):
        for img in image_data:
            if isinstance(img, dict):
                base64_str = img.get("data", "")
                if base64_str:
                    # Validation: check if base64_str already contains "data:image"
                    if base64_str.startswith("data:image"):
                        image_url = base64_str
                        log_debug(f"Using provided data:image URL for JSON image.")
                    else:
                        image_url = f"data:image/jpeg;base64,{base64_str}"
                        log_debug(f"Appended data:image/jpeg;base64, prefix to JSON image.")

                    user_content_items.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

    # 3. Add Images from native ComfyUI IMAGE tensor
    if tensor_images is not None:
        # Handle both torch.Tensor and numpy.ndarray
        is_torch = isinstance(tensor_images, torch.Tensor)
        is_numpy = isinstance(tensor_images, np.ndarray)
        
        if is_torch or is_numpy:
            # Process each image in the batch
            log_debug(f"Processing native ComfyUI IMAGE tensor batch. Type: {'torch' if is_torch else 'numpy'}, Shape: {tensor_images.shape}")
            # Determine batch dimension (first dimension for both torch and numpy)
            batch_size = tensor_images.shape[0]
            for i in range(batch_size):
                image_url = tensor_to_base64_data_uri(
                    tensor_images[i],
                    jpeg_quality=jpeg_quality,
                    max_dimension=max_dimension,
                    clear_memory=clear_memory,
                )
                if image_url:
                    user_content_items.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

    log_debug(f"Vision content array built with {len(user_content_items)} items total.")
    return user_content_items


def process_image_data_string(image_data_str: str) -> list:
    """Parse image_data string (JSON array) and return list of valid image elements."""
    if not image_data_str or not isinstance(image_data_str, str) or not image_data_str.strip():
        return []

    try:
        # Strip out newlines to ensure proper JSON parsing
        cleaned_str = image_data_str.replace("\n", "").replace("\r", "")
        images = json.loads(cleaned_str)
        if isinstance(images, list):
            log_debug(f"Successfully parsed image_data string into {len(images)} element list.")
            return images
    except json.JSONDecodeError as e:
        log_error(f"Error parsing JSON for parameter 'image_data': {image_data_str[:100]}...", e)
    return []
import base64
import gc
import io
import json
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    from logger import log_debug, log_error
except ImportError:
    from .logger import log_debug, log_error

# Image format magic numbers (file signatures)
IMAGE_MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
    b"RIFF....WEBP": "WEBP",
    b"BM": "BMP",
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
}

# Performance optimization constants
DEFAULT_JPEG_QUALITY = 85  # Balance between quality and file size
MAX_IMAGE_DIMENSION = 2048  # Maximum dimension for resizing large images
MIN_IMAGE_DIMENSION = 512  # Minimum dimension to maintain reasonable quality


def detect_image_format(data: Union[bytes, str]) -> Optional[str]:
    """Detect image format from magic numbers (file signatures).

    Args:
        data: Image data as bytes or base64 string

    Returns:
        Image format string (JPEG, PNG, GIF, WEBP, BMP, TIFF) or None if undetectable
    """
    if isinstance(data, str):
        try:
            data = base64.b64decode(data)
        except Exception:
            return None

    if not isinstance(data, bytes) or len(data) < 4:
        return None

    for magic, fmt in IMAGE_MAGIC_NUMBERS.items():
        if isinstance(magic, bytes) and data.startswith(magic):
            return fmt
        elif isinstance(magic, str):
            if data.startswith(magic.encode("utf-8")):
                return fmt

    return None


def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract metadata from an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing image metadata
    """
    try:
        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "aspect_ratio": img.width / img.height if img.height > 0 else 0,
                "has_transparency": img.mode in ("RGBA", "LA", "P"),
            }
    except Exception as e:
        log_error(f"Error extracting image metadata: {e}", e)
        return {}


def extract_tensor_metadata(tensor_image, batch_index: Optional[int] = None) -> Dict[str, Any]:
    """Extract metadata from a ComfyUI IMAGE tensor.

    Args:
        tensor_image: PIL Image or torch.Tensor or numpy array
        batch_index: Optional batch index if tensor_image is a batch

    Returns:
        Dictionary containing image metadata
    """
    try:
        # Convert tensor to numpy if needed
        if isinstance(tensor_image, torch.Tensor):
            tensor_image = tensor_image.cpu().numpy()

        # Convert to PIL Image
        i = 255.0 * tensor_image
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        return {
            "format": "PIL",
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "aspect_ratio": img.width / img.height if img.height > 0 else 0,
            "has_transparency": img.mode in ("RGBA", "LA", "P"),
            "batch_index": batch_index,
        }
    except Exception as e:
        log_error(f"Error extracting tensor metadata: {e}", e)
        return {}


def validate_image_data(image_data: str) -> bool:
    """Validate Base64 image data.

    Args:
        image_data: Base64 encoded image data

    Returns:
        True if valid image data, False otherwise
    """
    if not image_data or not isinstance(image_data, str):
        return False

    # Check for data URI format
    if image_data.startswith("data:image"):
        return True

    # Check if it's valid base64
    try:
        decoded = base64.b64decode(image_data)
        return len(decoded) > 0
    except Exception:
        return False


def tensor_to_base64_data_uri(
    tensor_image,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_dimension: Optional[int] = None,
    clear_memory: bool = True,
    output_format: str = "jpeg",
) -> str:
    """Convert ComfyUI image tensor to Base64 Data URI with performance optimizations.

    Args:
        tensor_image: PIL Image or torch.Tensor or numpy array
        jpeg_quality: JPEG quality (1-100), higher = better quality but larger file
        max_dimension: Maximum dimension to resize to (preserves aspect ratio)
        clear_memory: Whether to explicitly clear memory after processing
        output_format: Output format ('jpeg', 'png', 'webp')

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

        # Save to BytesIO with specified format
        buffered = io.BytesIO()

        if output_format.lower() == "png":
            # PNG supports transparency
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            img.save(buffered, format="PNG", optimize=True)
        elif output_format.lower() == "webp":
            # WebP supports transparency
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            img.save(buffered, format="WEBP", quality=jpeg_quality, optimize=True)
        else:
            # JPEG - convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)

        # Encode to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        log_debug(
            f"Converted ComfyUI IMAGE to Base64 Data URI. "
            f"Original Mode: {img.mode}, {output_format.upper()} Base64 length: {len(img_str)}, "
            f"Quality: {jpeg_quality}"
        )
        return f"data:image/{output_format};base64,{img_str}"
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
    extract_metadata: bool = False,
) -> Tuple[list, list]:
    """Build OpenAI Vision API compatible content array from text and images.

    Args:
        user_text: Text content for the user message
        image_data: List of image data objects from JSON parsing
        tensor_images: ComfyUI IMAGE tensor(s)
        jpeg_quality: JPEG quality for tensor image conversion
        max_dimension: Maximum dimension for resizing
        clear_memory: Whether to clear memory after processing
        extract_metadata: Whether to extract metadata from images

    Returns:
        Tuple of (list of content items, list of metadata dicts)
    """
    user_content_items = []
    metadata_list = []

    # 1. Add Text
    if user_text and user_text.strip():
        user_content_items.append({"type": "text", "text": user_text.strip()})

    # 2. Add Images from image_data (JSON strings)
    if isinstance(image_data, list):
        for idx, img in enumerate(image_data):
            if isinstance(img, dict):
                base64_str = img.get("data", "")
                if base64_str:
                    # Validation: check if base64_str already contains "data:image"
                    if base64_str.startswith("data:image"):
                        image_url = base64_str
                        log_debug("Using provided data:image URL for JSON image.")
                    else:
                        image_url = f"data:image/jpeg;base64,{base64_str}"
                        log_debug("Appended data:image/jpeg;base64, prefix to JSON image.")

                    user_content_items.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

                    # Extract metadata if requested
                    if extract_metadata:
                        metadata_list.append(
                            {
                                "source": "image_data",
                                "index": idx,
                                "format": "jpeg",
                                "size": len(base64_str),
                            }
                        )

    # 3. Add Images from native ComfyUI IMAGE tensor
    if tensor_images is not None:
        # Handle both torch.Tensor and numpy.ndarray
        is_torch = isinstance(tensor_images, torch.Tensor)
        is_numpy = isinstance(tensor_images, np.ndarray)

        if is_torch or is_numpy:
            # Process each image in the batch
            log_debug(
                "Processing native ComfyUI IMAGE tensor batch. "
                f"Type: {'torch' if is_torch else 'numpy'}, Shape: {tensor_images.shape}"
            )
            # Determine batch dimension (first dimension for both torch and numpy)
            batch_size = tensor_images.shape[0]
            for i in range(batch_size):
                # Extract metadata before converting to base64
                if extract_metadata:
                    metadata = extract_tensor_metadata(tensor_images[i], batch_index=i)
                    if metadata:
                        metadata_list.append(metadata)

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
    return user_content_items, metadata_list


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

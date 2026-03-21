import base64
import io
import json

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    from logger import log_debug, log_error
except ImportError:
    try:
        from .logger import log_debug, log_error
    except ImportError:
        try:
            from utils.logger import log_debug, log_error
        except ImportError:
            try:
                from utils.logger import log_debug, log_error
            except ImportError:
                from utils.logger import log_debug, log_error


def tensor_to_base64_data_uri(tensor_image):
    """Convert ComfyUI image tensor to Base64 Data URI."""
    try:
        # ComfyUI image tensors are typically (batch, height, width, channels)
        if isinstance(tensor_image, torch.Tensor):
            tensor_image = tensor_image.cpu().numpy()

        # Convert to PIL Image
        i = 255.0 * tensor_image
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Convert RGBA or other modes to RGB to avoid JPEG save errors
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to BytesIO as JPEG
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")

        # Encode to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        log_debug(
            f"Converted ComfyUI IMAGE to Base64 Data URI. Original Mode: {img.mode}, JPEG Base64 length: {len(img_str)}"
        )
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        log_error(f"Error converting image tensor to Base64", e)
        return None


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


def build_vision_content(user_text: str, image_data: list, tensor_images=None) -> list:
    """Build OpenAI Vision API compatible content array from text and images."""
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
        if isinstance(tensor_images, torch.Tensor):
            # Process each image in the batch
            log_debug(f"Processing native ComfyUI IMAGE tensor batch. Size: {tensor_images.shape}")
            for i in range(tensor_images.shape[0]):
                image_url = tensor_to_base64_data_uri(tensor_images[i])
                if image_url:
                    user_content_items.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

    log_debug(f"Vision content array built with {len(user_content_items)} items total.")
    return user_content_items

import base64
import io
import torch

from typing import Optional
from fastmcp import FastMCP
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

mcp = FastMCP("ocr-server")

_model     = None
_processor = None
_device    = None

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

TASK_PROMPTS = {
    "ocr"     : "OCR:",
    "table"   : "Table Recognition:",
    "formula" : "Formula Recognition:",
    "chart"   : "Chart Recognition:",
    "spotting": "Spotting:",
    "seal"    : "Seal Recognition:",
}


def load_model():
    global _model, _processor, _device

    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ocr-server] Loading {MODEL_ID} on {_device} ...")

    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
    ).to(_device).eval()

    print(f"[ocr-server] Model loaded on {_device}")


@mcp.tool
def ocr_image(image_base64: str, task: Optional[str] = "ocr") -> dict:
    """
    Run OCR on a base64-encoded image using PaddleOCR-VL-1.5.
    task options: ocr | table | formula | chart | spotting | seal
    """
    load_model()

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    task = task if task in TASK_PROMPTS else "ocr"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TASK_PROMPTS[task]},
            ],
        }
    ]

    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize              = True,
        return_dict           = True,
        return_tensors        = "pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=512)

    result = _processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])

    return {
        "raw_text": result,
        "task"    : task,
        "model"   : MODEL_ID,
        "device"  : _device,
    }


def run_server():
    load_model()
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)


if __name__ == "__main__":
    run_server()

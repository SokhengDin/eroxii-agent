import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


model_path  = "PaddlePaddle/PaddleOCR-VL-1.5"
image_path  = "test.jpeg"
task        = "ocr"

image = Image.open(image_path).convert("RGB")
orig_w, orig_h              = image.size
spotting_upscale_threshold  = 1500

if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
    process_w, process_h = orig_w * 2, orig_h * 2
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image = image.resize((process_w, process_h), resample_filter)


max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = {
    "ocr"   : "OCR:",
    "table" : "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart"     : "Chart Recognition:",
    "spotting"  : "Spotting:",
    "seal"  : "Seal Recognition:",
}

model       = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(DEVICE).eval()
processor   = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt = True,
#     tokenize        = True,
#     return_dict     = True,
#     return_tensors  = "pt",
#     images_kwargs   = {"size": {"shortest_edge": processor.image_processor.size, "longest_edge": max_pixels}},
# ).to(model.device)

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={
        "images_kwargs": {
            "size": {
                "shortest_edge": 448,
                "longest_edge": 1280
            }
        }
    }
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)
"""
main.py
----------
FastAPI inference API for Guns Object Detection using Faster R-CNN (ResNet-50 FPN).

This module launches a FastAPI server with two endpoints:
- GET  /           → Basic API health check.
- POST /predict/   → Accepts an uploaded image and returns the same image
                     annotated with detected gun bounding boxes.

Example
-------
Run the app locally:
    uvicorn src.main:app --reload

Access the interactive docs:
    http://localhost:8000/docs  → Swagger UI
"""

from __future__ import annotations

# -------------------------------------------------------------------
# Standard & Third-Party Imports
# -------------------------------------------------------------------
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw


# -------------------------------------------------------------------
# ⚙️ Model Setup
# -------------------------------------------------------------------

# Load pretrained Faster R-CNN with ResNet-50 FPN backbone
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set model to evaluation mode (disables dropout, batch norm updates)
model.eval()

# Detect available device — use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
model.to(device)

# Define transformation pipeline: convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])


# -------------------------------------------------------------------
# 🚀 FastAPI App Definition
# -------------------------------------------------------------------

# Create FastAPI instance with descriptive metadata
app = FastAPI(
    title="🔫 Guns Object Detection API",
    description="Upload an image to detect and annotate firearms using Faster R-CNN (ResNet-50 FPN).",
    version="1.0.0",
)


# -------------------------------------------------------------------
# 🧠 Prediction Utility Function
# -------------------------------------------------------------------
def predict_and_draw(image: Image.Image) -> Image.Image:
    """
    Run inference on an input image and draw bounding boxes for detected guns.
    """
    # Convert PIL image to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract boxes and confidence scores
    prediction = predictions[0]
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Convert to RGB and create drawing context
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # Draw boxes above confidence threshold (0.7)
    for box, score in zip(boxes, scores):
        if score > 0.7:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    # Return the annotated image
    return img_rgb


# -------------------------------------------------------------------
# 🌐 API Routes
# -------------------------------------------------------------------

# Root route for health check
@app.get("/")
def read_root() -> dict:
    # Return a simple welcome message
    return {"message": "Welcome to the Guns Object Detection API 🚀"}


# Prediction route for image uploads
@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Accept an uploaded image, perform inference, and return annotated results.
    """
    # Read raw image bytes from upload
    image_data = await file.read()

    # Open image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Run detection and draw bounding boxes
    output_image = predict_and_draw(image)

    # Convert annotated image to bytes
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Stream back the image in PNG format
    return StreamingResponse(img_byte_arr, media_type="image/png")
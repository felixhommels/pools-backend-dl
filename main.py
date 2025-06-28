from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import math
import uvicorn
import uuid
import base64

app = FastAPI()

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/best.pt")  # Your YOLO model path

def get_pixels_per_meter(lat: float, zoom: int) -> float:
    """
    Calculate the number of pixels per meter based on latitude and zoom level,
    assuming image is unscaled from Google Static Maps.
    """
    meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    return 1 / meters_per_pixel

@app.post("/api/detect-pools")
async def detect_pools(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    zoom: int = Form(...),
    address: str = Form(...)
):
    # Step 1: Read and prepare the image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize only if your model was trained on fixed-size images (e.g. 640x640)
    original_shape = img.shape[:2]  # (height, width)
    img = cv2.resize(img, (640, 640))

    # Step 2: Run YOLO prediction
    results = model.predict(img, conf=0.6)
    estimated_areas_m2 = []

    # Step 3: Calculate ppm
    ppm = get_pixels_per_meter(latitude, zoom)

    # Step 4: Loop over results and calculate real-world area
    for result in results:
        # Case 1: Segmentation masks
        if hasattr(result, "masks") and result.masks:
            for mask in result.masks.xy:
                polygon = np.array(mask, dtype=np.float32)
                pixel_area = cv2.contourArea(polygon)
                real_area = pixel_area / (ppm ** 2)
                estimated_areas_m2.append(real_area)

        # Case 2: Oriented Bounding Boxes (OBB)
        elif hasattr(result, "obb") and result.obb is not None:
            for polygon in result.obb.xyxy:
                polygon_np = polygon.cpu().numpy().tolist()
                contour = np.array(polygon_np, dtype=np.float32)
                pixel_area = cv2.contourArea(contour)
                real_area = pixel_area / (ppm ** 2)
                estimated_areas_m2.append(real_area)

        # Case 3: Classic bounding boxes
        elif result.boxes is not None:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                pixel_area = (x2 - x1) * (y2 - y1)
                real_area = pixel_area / (ppm ** 2)
                estimated_areas_m2.append(real_area)

    annotated_img = img.copy()

    for result in results:
        boxes = result.boxes
        masks = result.masks

        if boxes is not None:
            for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls.item())
                confidence = float(conf.item())
                label = f"{model.names[class_id]} {confidence:.2f}"

                # Polygon mask (shaded area) 
                if masks is not None and masks.xy is not None and i < len(masks.xy):
                    polygon = np.array(masks.xy[i], dtype=np.int32)
                    overlay = annotated_img.copy()
                    cv2.fillPoly(overlay, [polygon], color=(255, 0, 255))  # BGR pink/rosa
                    cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)

                # Bounding box 
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Label 
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + text_width + 4, y1), (255, 0, 0), -1)
                cv2.putText(annotated_img, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(f"annotated_outputs/{filename}", annotated_img)
    
    _, buffer = cv2.imencode(".jpg", annotated_img)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    
    # Step 6: Return response
    if estimated_areas_m2:
        return JSONResponse({
            "estimated_areas_m2": estimated_areas_m2,
            "ppm": ppm,
            "image_shape": original_shape,
            "image_base64": encoded_img
        })
    else:
        return JSONResponse({"message": "No pools detected."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
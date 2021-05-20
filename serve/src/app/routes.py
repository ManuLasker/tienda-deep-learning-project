from typing import Any, Dict, Tuple
from fastapi import Depends, HTTPException
from starlette.responses import RedirectResponse
from app import app
from app.dl_model import health_check_models
from app.route_model import InvocationRequest, InvocationResponse, PingResponse
from app.dl_model.image import YoloInput


@app.get("/", include_in_schema=False)
def docs_redirect():
    import os

    prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").strip("/")
    return RedirectResponse(prefix + "/docs")


@app.get("/ping", response_model=PingResponse)
def ping(models_check: Tuple[bool, bool] = Depends(health_check_models)):
    """Health check endpoint used by sagemaker. This needs to return 200,
    to notify sagemaker that the server is all good.
    """
    models_names = [
        "YoloV5Predictor_loading_Status",
        "ClassifierPredictor_loading_status",
    ]
    models_info = dict(zip(models_names, models_check))
    if not all(models_check):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model loading error",
                "message": "models were not loaded correctly!",
                "models_info": models_info,
            },
        )
    return {
        "message": "health check passed: all models where already loaded.",
        "models_info": models_info,
    }


@app.post("/invocations", response_model=InvocationResponse)
def invocation(
    body: InvocationRequest,
    models_check: Tuple[bool, bool] = Depends(health_check_models),
):
    """Invocation endpoint used by sagemaker, this will execute the yolov5 and classification on an image
    in base64 format and return the detected products.
    """
    try:
        # Create yolo input object
        yolo_input = YoloInput.from_base64_image(
            body.base64_image, new_shape=(224, 224)
        )
        # Detect objects
        return yolo_input.detect_products(conf_thres=0.1)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "There was an error on the decoding of body payload or in the detections"
                            ", check your encode base64 image is well encoded",
            },
        )

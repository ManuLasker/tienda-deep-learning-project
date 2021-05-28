from pydantic import BaseModel
from typing import Tuple, List


class DetectedProduct(BaseModel):
    """Detected product model, this contains information
    about the detected product such as product name and the bounding box
    dimensions as follow:
    {
        product_id: id
        product_name: str,
        detection_index: int,
        confidence: float,
        bounding_box: [x0, y0, x1, y1]
        top_k_product_names: Tuple[str, str, str]
        top_k_confidences: Tuple[float, float, float]
        top_k_product_ids: Tuple[int, int, int]
        top_k_detection_indices: Tuple[int, int, int]
    }
    """

    product_id: int
    product_name: str
    detection_index: int
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    top_k_product_names: Tuple[str, str, str]
    top_k_confidences: Tuple[float, float, float]
    top_k_product_ids: Tuple[int, int, int]
    top_k_detection_indices: Tuple[int, int, int]

class InvocationRequest(BaseModel):
    """Invocation Request for /invocations endpoint,
    This must contain an image in base64 string format encoded as follow
    {
        base64_image: str
    }
    """

    base64_image: str  # input image


class InvocationResponse(BaseModel):
    """Invocation Response for /invocations endpoint,
    this will respond an image in base64 string format encoded, and
    the detected products information as follows:
    {
        total_classes: int
        detected_products: [
            {
                product_id: id
                product_name: str,
                detection_index: int,
                confidence: float,
                bounding_box: [x0, y0, x1, y1]
                top_k_product_names: Tuple[str, str, str]
                top_k_confidences: Tuple[float, float, float]
                top_k_product_ids: Tuple[int, int, int]
                top_k_detection_indices: Tuple[int, int, int]
            }
        ]
    }
    """

    total_classes: int
    detected_products: List[DetectedProduct]


class ModelInfo(BaseModel):
    """Model loading status information.
    {
        YoloV5Predictor_loading_Status: bool,
        ClassifierPredictor_loading_status: bool
    }
    """

    YoloV5Predictor_loading_Status: bool
    ClassifierPredictor_loading_status: bool


class PingResponse(BaseModel):
    """ping response for /ping endpoint.
    this will respond status model or raise an exception.
    {
        message: str,
        models_info: {
            YoloV5Predictor_loading_Status: bool,
            ClassifierPredictor_loading_status: bool
        }
    }
    """

    message: str
    models_info: ModelInfo

from pydantic import BaseModel
from typing import Tuple, List
from typing_extensions import TypedDict

class DetectedProduct(TypedDict):
    """Detected product type dictionary, this contains information
    about the detected product such as product name and the bounding box
    dimensions as follow:
    {
        product_name: str,
        bounding_box: [x0, y0, x1, y1]
    }
    """
    product_name: str
    bounding_box: Tuple[int, int, int, int]

class InvocationRequest(BaseModel):
    """Invocation Request for /invocations endpoint,
    This must contain an image in base64 string format encoded as follow
    {
        image: str
    }
    """
    image: str # input image

class InvocationResponse(BaseModel):
    """Invocation Response for /invocations endpoint,
    this will respond an image in base64 string format encoded, and 
    the detected products information as follows:
    {
        image: str,
        detected_products: {
            product_name: str,
            bounding_box: [x0, y0, x1, y1]
        }
    }
    """
    image: str # output image with bounding boxes
    detected_products: List[DetectedProduct]    
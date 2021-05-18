from typing import Any
from app import app
from app.route_model import InvocationRequest, InvocationResponse

@app.get('/ping')
def ping():
    """Health check endpoint used by sagemaker. This needs to return 200,
    to notify sagema that the server is all good.
    """
    return {'message':'all is well'}

@app.post('/invocations', response_model=InvocationResponse)
def invocation(body: InvocationRequest) -> Any:
    """Invocation endpoint used by sagemaker, this will execute the yolov5 model and return the detected products
    Args:
        body (InvocationRequest): Invocation response 
    Returns:
        Any: Invocation response will return the original image and the detected products
        with information about the dimensions of the bounding boxes
    """
    dummy_response = {
        'image': 'hola',
        'detected_products': [{'product_name': 'papitas_margarita',
                               'bounding_box': (1,2,3,4)}]
    }
    return dummy_response
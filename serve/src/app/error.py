from app import app
from fastapi import Request
from fastapi.responses import JSONResponse

class ModelLoadingError(Exception):
    """Model loading exception exteds from exception python class.
    """
    def __init__(self, detail:str):
        self.detail = detail

@app.exception_handler(ModelLoadingError)
def model_loading_exception_handler(request: Request, exception: ModelLoadingError):
    """handling model loading error response.

    Args:
        request (Request): fastapi request object.
        exception (ModelLoadingError): model loading exception raised.
    """
    return JSONResponse(
        status_code=500,   
        content={"message": f"There was an error while loading models",
                 "detail": exception.detail}
    )
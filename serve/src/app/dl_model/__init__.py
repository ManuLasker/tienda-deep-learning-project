import os
import json
from typing import Generator, Tuple
from app.dl_model.model.predictor import YoloV5Predictor, ClassifierPredictor
from app.dl_model import YoloV5Predictor, ClassifierPredictor
from app.error import ModelLoadingError

# Method for telling the app that models where loaded
def health_check_models() -> Generator[None, Tuple[bool, bool], None]:
    """Health check for models loaded to app.

    Raises:
        ModelLoadingError: raise model error loading if error catches
        
    Yields:
        [Generator[bool, bool]]: generator indicating if model where loaded.
    """
    try:
        # setup predictors before creating the app
        YoloV5Predictor.setup_model(
            model_path="/opt/ml/model/"+os.environ.get("YOLO_MODEL_NAME"),
            class_names=json.loads(os.environ.get("YOLO_CLASSES_NAMES")),
            anchors=json.loads(os.environ.get("ANCHORS"))
        )

        ClassifierPredictor.setup_model(
            model_path="/opt/ml/model/"+os.environ.get("CLASSIFICATION_MODEL_NAME"),
            class_names=json.loads(os.environ.get("CLASSIFICATION_CLASSES_NAMES")),
            product_external_ids=json.loads(os.environ.get("PRODUCT_EXTERNAL_IDS"))
        )
        yield (YoloV5Predictor.model is not None,
               ClassifierPredictor.model is not None)
    except Exception as exception:
        raise ModelLoadingError(detail=str(exception))
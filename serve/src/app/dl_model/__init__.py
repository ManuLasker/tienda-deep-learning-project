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
            model_path="/opt/ml/model/224_mvp.pt",
            class_names=["items"],
            anchors = [
                [10,13, 16,30, 33,23], # P3/8
                [30,61, 62,45, 59,119], # P4/16
                [116,90, 156,198, 373,326] # P5/32
            ]
        )

        ClassifierPredictor.setup_model(
            model_path="/opt/ml/model/mobilenetv3.pt",
            class_names=["Arroz Doble Vitamor Diana x 500 g",
                        "CocaCola x 250 ml",
                        "Maracuya",
                        "Chicharrón Americano Jacks x 15 g",
                        "CocaCola x 400 ml",
                        "Papas de limón 39gr"],
            product_external_ids=[29856, 30978, 31742, 32057, 30981, 32191]
        )
        yield (YoloV5Predictor.model is not None,
               ClassifierPredictor.model is not None)
    except Exception as exception:
        raise ModelLoadingError(detail=str(exception))
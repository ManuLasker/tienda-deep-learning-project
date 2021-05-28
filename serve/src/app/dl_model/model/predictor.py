import torch
from collections import namedtuple
from typing import Dict, List, Tuple
from app.dl_model.model.base_predictor import BasePredictor

class YoloV5Predictor(BasePredictor):
    """YoloV5 predictor class inherited from PredictorBase
    """
    # Class Attributes
    model_path = None
    model = None
    class_names = None
    anchors = None
    
    def __init__(self):
        super().__init__()
        assert self.model is not None, "Predictor was not set properly"
        
    @classmethod
    def get_stride(cls) -> torch.Tensor:
        """Generate stride for the model to post process predictions

        Returns:
            torch.Tensor: stride tensor
        """
        return torch.tensor([128/x.shape[-2]
                             for x in cls.model(torch.zeros(1, 3, 128, 128))])
        
    @classmethod
    def setup_model(cls, model_path:str, class_names: List[str],
                    anchors: List[List[int]]) -> None:
        """Setup model for inference.

        Args:
            model_path (str): str /path/to/model.
            class_names (List[str]): List of class names for detected products.
            anchors (List[List[int]]): List of anchors given by model trained configs.
        """
        if not cls.model:
            # number of classes 
            cls.nc = len(class_names)
            # number of layers
            cls.nl = len(anchors)
            # number of outputs per anchors
            cls.no = cls.nc + 5
            # set up anchors and class_names for class
            cls.class_names = class_names
            anchors_temp = torch.tensor(anchors, dtype=torch.float32).view(cls.nl, -1, 2)
            cls.anchors = anchors_temp.clone()
            # set model_path and load model
            cls.set_model_path(model_path)
            cls.load_model()
            # set up the model configuration for post processing prediction
            cls.anchor_grid = anchors_temp.clone().view(cls.nl, 1, -1, 1, 1, 2)
            cls.stride = cls.get_stride()
            cls.anchors /= cls.stride.view(-1, 1, 1)
            # check anchors order
            cls.check_anchor_order()
        
    @classmethod
    def check_anchor_order(cls):
        """Check anchor order against stride order for yolov5 Detect() module, and 
        correct if necessary
        """
        a = cls.anchor_grid.prod(-1).view(-1) # anchor area
        da = a[-1] - a[0] # delta a
        ds = cls.stride[-1] - cls.stride[0] # delta s
        if da.sign() != ds.sign(): # same order
            print('Reversing anchor order')
            cls.anchors[:] = cls.anchors.flip(0)
            cls.anchor_grid[:] = cls.anchor_grid.flip(0)
            
    @staticmethod
    def _make_grid(nx:int=20, ny:int=20) -> torch.Tensor:
        """Makes grid for post processing yolov5 Detect Head

        Args:
            nx (int, optional): number of x points. Defaults to 20.
            ny (int, optional): number of y points. Defaults to 20.

        Returns:
            torch.Tensor: stack of meshgrid between arange of nx and ny points.
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()        
        
    def predict(self, image_tensor:torch.Tensor,
                **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        """get predictions of image tensor of shape [C, H, W].

        Args:
            image_tensor (torch.Tensor): Image tensor in RGB format, with dimensions
                                        [Nc, Height, Width], where Nc is the number of channels.
            **kwargs (kwargs): Keyword arguments to pass to prediction post process output method,
                        Process head output for yolov5 model and apply non max suppression to
                        get detections with shape: [nx6 (x1, y1, x2, y2, conf, class_name)].

        Returns:
            [List[Tuple[float, float, float, float, float, str]]]: list of detections with dimensions of
                [nx6 (x1, y1, x2, y2, conf, class_name)]
        """
        import app.dl_model.model.prediction as prediction
        self.model.eval()
        with torch.no_grad():
            preds = self.model(image_tensor.unsqueeze(0))
        return prediction.YoloV5Prediction(self, output_tensor=preds).get_tensor_detections(**kwargs)
    
    
class ClassifierPredictor(BasePredictor):
    """Classifier Predictor to handle classification model.
    """
    # Class Attributes
    model_path = None
    model = None
    class_names = None
    product_external_ids = None
    
    def __init__(self):
        super().__init__()
        assert self.model is not None, "Predictor was not set properly"
        
    def predict(self, image_tensor:torch.Tensor):
        """Predict class producto for image tensor

        Args:
            image_tensor (torch.Tensor): image tensor of shape [1, c, h, w]

        Returns:
            [ClassifierPrediction]: classifier prediction namedtuple with 
                properties product_id, class_name, conf, detection_index,
                top_k_names, top_k_indices, top_k_product_id, top_k_confidences.
        """
        self.model.eval()
        with torch.no_grad():
            preds:torch.Tensor = self.model(image_tensor.unsqueeze(0))
        return self.post_process(preds.softmax(1))
    
    def post_process(self, preds: torch.Tensor):
        """post process the predictions of the predictor to a dictionary
        with keys 'class_name':, 'conf': 

        Args:
            preds (torch.Tensor): torch Tensor prediction with shape [1, number_classes].

        Returns:
            [ClassifierPrediction]: NamedTuple containing the information about the prediction.
        """
        ClassifierPrediction = namedtuple("ClassifierPrediction",
                                         ["product_id", "class_name",
                                          "conf", "detection_index",
                                          "top_k_names", "top_k_indices",
                                          "top_k_confidences", "top_k_product_ids"])
        top_k = 3 # top k predictions values
        # Using topk to get topk values and indices for predictions
        conf_values, index_values = map(lambda x: x.view(-1), preds.topk(top_k, dim=1))
        conf, index = conf_values[0], index_values[0]
        
        return ClassifierPrediction(product_id=self.product_external_ids[index.item()],
                                    class_name=self.class_names[index.item()],
                                    conf=conf.item(),
                                    detection_index=index.item(),
                                    top_k_names=[self.class_names[indx] for indx in index_values],
                                    top_k_indices=index_values.tolist(),
                                    top_k_confidences=conf_values.tolist(),
                                    top_k_product_ids=[self.product_external_ids[indx] 
                                                       for indx in index_values])

    @classmethod
    def setup_model(cls, model_path: str,
                    class_names: List[str], 
                    product_external_ids: List[int]) -> None:
        """Set up classification model for prediction.

        Args:
            model_path (str): str /path/to/model.
            class_names (List[str]): List of class names for classification model.
            product_external_ids (List[int]): List of products ids for database.
                class names and product external ids must match, product_id: product_name:
        """
        if not cls.model:
            # set model_path and load model
            cls.set_model_path(model_path)
            cls.load_model()
            # set Attributes
            cls.class_names = class_names
            cls.product_external_ids = product_external_ids
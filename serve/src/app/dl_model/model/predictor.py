import torch

from typing import List
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
        
    def predict(self, image_tensor:torch.Tensor):
        import app.dl_model.model.prediction as prediction
        self.model.eval()
        with torch.no_grad():
            pred = self.model(image_tensor.unsqueeze(0))
        return prediction.YoloV5Prediction(self, output_tensor=pred).processed_output
    
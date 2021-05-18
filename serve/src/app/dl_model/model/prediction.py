import torch
import app.dl_model.model.predictor as predictor
from typing import List

class YoloV5Prediction:
    """Yolo Predction Class to handle yolov6 model prediction.
    """
    
    def __init__(self, model: predictor.YoloV5Predictor,
                 output_tensor: List[torch.Tensor]):
        self.__model = model
        self.output_tensor = output_tensor
    
    @property
    def processed_output(self):
        """Process Detect head output for yolov5 model.
        """
        z = []
        grid = [None] * self.__model.nl
        for i in range(self.__model.nl):
            bs, _, ny, nx, _ = self.output_tensor[i].shape
            grid[i] = self.__model._make_grid(nx, ny)
            y = self.output_tensor[i].sigmoid()
            y[:,:,:,:, 0:2] = ((y[:,:,:,:, 0:2] * 2 - 0.5 + grid[i]) 
                               * self.__model.stride[i]) # xy
            y[:,:,:,:, 2:4] = ((y[:,:,:,:, 2:4] * 2) ** 2 
                               * self.__model.anchor_grid[i]) # wh
            z.append(y.view(bs, -1, self.__model.no))
        return torch.cat(z, 1)
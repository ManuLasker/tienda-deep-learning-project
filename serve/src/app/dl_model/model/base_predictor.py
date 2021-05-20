import torch
from abc import abstractmethod, ABC

class BasePredictor(ABC):
    """Predictor Base Class for deep learning models
    """
    model:torch.nn.Module = None
    model_path:str = None
    
    @classmethod
    def load_model(cls):
        if cls.model is None:
            cls.model = torch.jit.load(cls.model_path)
    
    @classmethod
    def set_model_path(cls, path:str):
        if cls.model_path is None:
            cls.model_path = path
            
    @abstractmethod
    def predict(self, *args, **kwargs):
        ...
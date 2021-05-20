import torch
import numpy as np
import app.dl_model.model.predictor as predictor
from typing import Tuple, Union, List, Dict, Any
from app.dl_model.utils import ImageUtilities
from app.dl_model.detection import DetectedObject, DetectedObjects

class YoloInput:
    def __init__(self, numpy_image: np.ndarray, new_shape: Tuple[int, int]) -> None:
        """Initialize yolo input object.
        Args:
            numpy_image (np.ndarray): numpy original image.
            new_shape (Tuple[int, int]): new shape for preprocessing step.
        """
        self.numpy_image = numpy_image # numpy image [H x W x C]
        self.new_shape = new_shape
        self.original_size = numpy_image.shape[:-1] # [H x W]
        self.already_detected = False

    @classmethod
    def from_file(cls, file_path: str,
                  new_shape: Union[Tuple[int, int], int] = (224, 224)) -> 'YoloInput':
        """Create a yolo input object from a given image file
        
        Args:
            file_path (str): /path/to/imagefile.
            new_shape (Union[Tuple[int, int], int]): new shape for preprocessing.

        Returns:
            [YoloInput]: yolo input object. with image and new size.
        """
        # Load BGR numpy image array
        numpy_image = ImageUtilities._load_image_from_file(file_path)
        # validate new_size 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # return yolo input object
        return cls(numpy_image, new_shape)
    
    @classmethod
    def from_base64_image(cls, base64_image: str,
                          new_shape: Union[Tuple[int, int], int] = (224, 224)) -> 'YoloInput':
        """Create a yolo input object from a base64 encoded string image.
        
        Args:
            base64_image (str): base64 encoded string image.
            new_shape (Tuple[int, int]): new size for preprocessing.
            
        Returns:
            [YoloInput]: yolo input object. with image and new size.
        """
        # Load BGR numpy image array
        numpy_image = ImageUtilities._load_image_from_base64_image(base64_image)
        # validate new_size 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # return yolo input object
        return cls(numpy_image, new_shape)
    
    def detect_products(self, conf_thres: float = 0.25,
                              iou_thres: float = 0.45,
                              classes: List[str] = None,
                              agnostic: bool = False,
                              multi_label: bool = False,
                              second_classifier: bool = False) -> Dict[str, Any]:
        """Apply detection yolo model for input yolo, and if second classsifier
        is True then apply second classifier to yolo detections.

        Args:
            conf_thres (float, optional): confidence threshold for nms algorithm. Defaults to 0.25.
            iou_thres (float, optional): intersection or union threshold for nms algorithm. Defaults to 0.45.
            classes (List[str], optional): class list to filter by classes. Defaults to None.
            agnostic (bool, optional): to add agnostic. Defaults to False.
            multi_label (bool, optional): habilitate multiple labels per box. Defaults to False.
        """
        # Instantiate yolo model predictor.
        self.model = predictor.YoloV5Predictor()
        # preprocess yolo input data
        tensor_input_image = self.preprocess()
        # get detections from the model this will return us box with 
        predictions = self.model.predict(tensor_input_image)
        # Create detect products object list
        detected_products = DetectedObjects(ImageUtilities._transform_image(self.numpy_image))
        # add detected object
        for pred in predictions:
            coords = pred[:4]
            conf, class_name = pred[4:] 
            detected_products.add_detected_object(DetectedObject(self.original_size,
                                                                 self.resize_shape,
                                                                 coords, conf, class_name))
        # apply classifier
        detected_products.apply_classifier()
        # save to class
        self.detected_products = detected_products
        self.already_detected = True
        return detected_products.json()
        
    def preprocess(self) ->  torch.Tensor:
        """Preprocess yolo input numpy image to use predictor

        Returns:
            torch.Tensor: tensor image preprocessed.
        """
        # Apply rectangular resize to numpy image BGR.
        numpy_imager = ImageUtilities.rectangular_resize_with_pad(self.numpy_image,
                                                                  new_shape=self.new_shape)
        # create resize shape
        self.resize_shape = numpy_imager.shape[:-1] #(H, W)
        # Transform to RGB format
        numpy_imager = ImageUtilities._transform_image(numpy_imager, "RGB")
        # return tensor normalize to [0 - 1], RGB format and [C, H, W] dimensions.
        return ImageUtilities._to_tensor(numpy_imager)

class ClassifierInput:
    def __init__(self, numpy_image: np.ndarray, new_shape: Tuple[int, int]):
        """Constructor for ClassifierInput object, this is used to 
        classify a specific numpy image.

        Args:
            numpy_image (np.ndarray): numpy image array.
            new_shape (Tuple[int, int]): new size for preprocessing step.
        """
        self.numpy_image = numpy_image
        self.new_shape = new_shape
    
    def preprocess(self) -> torch.Tensor:
        """preprocess classifier input numpy image to use predictor,
        this preprocessing is fastai preprocessing with mean and std from imagenet, 
        and non keep aspect ratio resize method.

        Returns:
            torch.Tensor: tensor image preprocessed
        """
        # Apply fastai preprocessing to numpy image
        return ImageUtilities.process_fastai_model(self.numpy_image, self.new_shape)
    
    def predict_class(self):
        """classify image input
        """
        # preprocess numpy imput image
        tensor_input_image = self.preprocess()
        # Instantiate predictor
        self.model = predictor.ClassifierPredictor()
        # predict
        return self.model.predict(tensor_input_image)
    
    @classmethod
    def get_total_classes(cls) -> int:
        return len(predictor.ClassifierPredictor.class_names)
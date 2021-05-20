from PIL.Image import new
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from enum import Enum

class BoundingBoxFormat(str, Enum):
    """Enum class for bounding box formats.
    """
    xyxy='xyxy' # format for top-left and bottom-right
    xywh='xywh' # format for top-left and width and height
    xmymwh='xmymwh' # format for center and width and height


class DetectedObject:
    def __init__(self, original_shape: Tuple[int, int],
                 new_shape: Tuple[int, int],
                 coordinates: Tuple[float, float, float, float],
                 conf: float,
                 class_name: str,
                 bb_format: BoundingBoxFormat = BoundingBoxFormat.xyxy):
        self.original_shape = original_shape
        self.new_shape = new_shape
        self.coordinates:np.ndarray = np.array([coordinates])
        self.conf = conf
        self.class_name = class_name
        self.bb_format = bb_format
        
    def scale_up_coordinates(self) -> None:
        self.scale_coordinates = BoxCoordsUtilities.scale_coords(self.new_shape,
                                                           self.coordinates,
                                                           self.original_shape).reshape(-1)
    
    def predict_class(self, original_image_numpy: np.ndarray) -> None:
        """Predict class for each Detecte object. This will update class names
        and set class id if it was given.

        Args:
            original_image_numpy (np.ndarray): numpy array in RGB format 
                to get classification predictor.
        """
        from app.dl_model.image import ClassifierInput
        # scale up coordinates
        self.scale_up_coordinates()
        x1, y1, x2, y2 = [int(coord) for coord in self.scale_coordinates.round()]
        # crop original numpy image
        numpy_image = original_image_numpy[y1:y2, x1:x2, :].copy()
        # create classifier input object
        classifier_input = ClassifierInput(numpy_image, new_shape=(224, 224))
        # classify input
        prediction = classifier_input.predict_class()
        # set attributes
        self.class_name = prediction.class_name # update class_name
        self.conf = prediction.conf # update probability
        self.product_id = prediction.product_id # set product external id
        self.detection_index = prediction.detection_index
        
    def json(self) -> Dict[str, Any]:
        """Return a nice format for detected object

        Returns:
            Dict[str, Any]: json format for detected object
        """
        return {
                    "product_id": self.product_id,
                    "detection_index": self.detection_index,
                    "product_name": self.class_name,
                    "confidence": self.conf,
                    "bounding_box": [int(coord) for coord in self.scale_coordinates.round()]
                }


class DetectedObjects:
    def __init__(self, original_image_numpy: np.ndarray):
        """Constructor for detected objects

        Args:
            original_image_numpy (np.ndarray): original numpy image in RGB format
        """
        self.original_image = original_image_numpy
        self.detected_objects:List[DetectedObject] = []

    def add_detected_object(self, detected_object: DetectedObject):
        """add detected object to the list of detected objects

        Args:
            detected_object (DetectedObject): detected object to add
        """
        self.detected_objects.append(detected_object)
        
    def apply_classifier(self):
        """Apply classifier to each detected object
        """
        for detected_object in self.detected_objects:
            detected_object.predict_class(self.original_image)
            
    def json(self) -> Dict[str, List]:
        """return all the detections in a nice json format

        Returns:
            List[str, List]: list of detected products
        """
        from app.dl_model.image import ClassifierInput
        return {
            "total_classes": ClassifierInput.get_total_classes(),
            "detected_products": [detected_object.json() for detected_object in self.detected_objects]
            }


class BoxCoordsUtilities:
    
    @staticmethod
    def xmymwh2xyxy(coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert coordinates of box from format (center x, center y, width, height) to format
        (x1, y1, x2, y2) where xy1=top-left, xy2=bottom-right
        Args:
            coords (Union[np.ndarray, torch.Tensor]): numpy or tensor coordinates
                in format (center x, center y, width, height) with dimensions (?, 4)

        Returns:
            Union[np.ndarray, torch.Tensor]:  numpy or tensor coordinates in 
                format (x1, y1, x2, y2)
        """
        new_coords = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
        new_coords[:, 0] = coords[:, 0] - coords[:, 2] / 2  # top left x
        new_coords[:, 1] = coords[:, 1] - coords[:, 3] / 2  # top left y
        new_coords[:, 2] = coords[:, 0] + coords[:, 2] / 2  # bottom right x
        new_coords[:, 3] = coords[:, 1] + coords[:, 3] / 2  # bottom right y
        return new_coords
    
    @staticmethod
    def clip_coords(coords: Union[np.ndarray, torch.Tensor],
                    img_shape: Tuple[int, int]) -> Union[np.ndarray, torch.Tensor]:
        """clip coords to image shape

        Args:
            coords (Union[np.ndarray, torch.Tensor]): numpy or tensor array coordinates.
                with dimensions (?, 4)
            img_shape (Tuple[int, int]): img shape (H, W)

        Returns:
            Union[np.ndarray, torch.Tensor]: return clipped coordinates
        """
        new_coords = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
        clip = (lambda x, index: torch.clamp(x, 0, img_shape[index]) if isinstance(x, torch.Tensor)
                                else np.clip(x, 0, img_shape[index]))
        for i in range(4):
            new_coords[:, i] = clip(new_coords[:, i], (i + 1)%2)
        return new_coords
    
    @staticmethod
    def xyxy2xmymwh(coords:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """ Convert nx4 boxes coordinates from xyxy format to xmymwh format.
        (x1, y1, x2, y2) where xy1=top-left, xy2=bottom-right
        (xm, ym, w, h) x-center y-center, width, height.
        Args:
            coords (Union[np.ndarray, torch.Tensor]): coordinates.

        Returns:
            Union[np.ndarray, torch.Tensor]: transformed coordinates.
        """
        new_coords = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
        new_coords[:, 0] = (coords[:, 0] + coords[:, 2]) / 2  # x center
        new_coords[:, 1] = (coords[:, 1] + coords[:, 3]) / 2  # y center
        new_coords[:, 2] = coords[:, 2] - coords[:, 0]  # width
        new_coords[:, 3] = coords[:, 3] - coords[:, 1]  # height
        return new_coords
    
    @staticmethod
    def scale_coords(img1_shape: Tuple[int, int],
                     coords: Union[np.ndarray, torch.Tensor],
                     img0_shape: Tuple[int, int]) -> Union[np.ndarray, torch.Tensor]:
        """Reescale coords (x, y, x, y) format from img1_shape 
        to img0_shape. coordinates needs to be in xyxy format.

        Args:
            img1_shape (Tuple[int, int]): img from, shape in (H, W)
            coords (Union[np.ndarray, torch.Tensor]): coordinates tensor or numpy array 
                    with dimensions (?, 4)
            img0_shape (Tuple[int, int]): img to, shape in (H, W)

        Returns:
            Union[np.ndarray, torch.Tensor]: new coordinates.
        """
        # calculate gain and padding
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1]) # gain = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2) # wh padding
        new_coords = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
        new_coords[:, [0, 2]] -= pad[0] # x padding
        new_coords[:, [1, 3]] -= pad[1] # y padding
        new_coords[:, :4] /= gain # escale
        return BoxCoordsUtilities.clip_coords(new_coords, img0_shape)
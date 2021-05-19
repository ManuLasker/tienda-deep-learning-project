import torch
import torchvision
import app.dl_model.model.predictor as predictor
from typing import List, Tuple
from app.dl_model.detection import BoxCoordsUtilities

class YoloV5Prediction:
    """Yolo Predction Class to handle yolov6 model prediction.
    """
    
    def __init__(self, model: predictor.YoloV5Predictor,
                 output_tensor: List[torch.Tensor]):
        self.__model = model
        self.output_tensor = output_tensor
    
    def get_tensor_detections(self, **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        """Process Detect head output for yolov5 model and apply non max suppression to
        get detections with shape: [nx6 (x1, y1, x2, y2, conf, class_name)].
        
        Args:
            **kwargs (kwargs): Keyword arguments to pass to non max suppression algorithm and
                            get list of detections with shape nx6 (x1, y1, x2, y2, conf, class_name)
        Returns:
            [List[Tuple[float, float, float, float, float, str]]]: list of detections
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
        prediction = torch.cat(z, 1)
        return self.non_max_suppression(prediction, **kwargs)
    
    def non_max_suppression(self, prediction: torch.Tensor,
                            max_det: int = 300,
                            **kwargs) -> List[Tuple[float, float, float, float, float, str]]:
        """Apply non max supression algorithm to given yolo processed output.

        Args:
            prediction (torch.Tensor): tensor predictions with dimensions [1, anchors_prod, no],
                        where no is the number of outputs per anchors.
            max_det (int, optional): max detection threshold. Defaults to 300.

        Returns:
            [List[Tuple[float, float, float, float, float, str]]]: list of detections
        """
        import time
        # Get arguments from keywords to validate.
        conf_thres: float = kwargs.get('conf_thres', 0.25)
        iou_thres: float = kwargs.get('iou_thres', 0.45)
        classes: list[str] = kwargs.get('classes', None)
        agnostic: bool = kwargs.get('agnostic', False)
        multi_label: bool = kwargs.get('multi_label', False)
        
        # Begin procedure
        nc = prediction[0].shape[1] - 5 # get number of classes
        xc = prediction[..., 4] > conf_thres # filter candidates by confidence
        
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        
        t = time.time()
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction): # image index, image inference this will jus be one cicle
            x:torch.Tensor = x[xc[xi]] # select the confidence ones
            
            # if none remain process next image
            if not x.shape[0]:
                continue
            
            # compute conf
            x[:, 5:] *= x[:, 4:5] # conf = obj_conf * cls_conf
            
            # Transform  box from format (center x, center y, width, height) to format
            # (x1, y1, x2, y2)
            box = BoxCoordsUtilities.xmymwy(coords = x[:, :4])
            
            # Detections matrix nx6 (xyxy, conf, cls index)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else: # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                
            # filter by classes
            if classes is not None:
                try:
                    classes_index = [self.__model.class_names.index(class_name)
                                     for class_name in classes]
                    x = x[(x[:, 5:6] == torch.tensor(classes_index, device=x.device)).any(1)]
                except ValueError:
                    print("Filter by class name will not work properly!,"
                          "one of  the class names does not exist in {}".format(self.__model.class_names))
                    
            # check shape
            n = x.shape[0] # number of boxes
            if not n: # no boxes
                continue
            elif n > max_nms: # excees boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
                
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            
            output[xi] = x[i].tolist()
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded
            def change_classIndex_by_name(out: List[float]):
                out[-1] = self.__model.class_names[int(out[-1])]
                return out
        return list(map(change_classIndex_by_name, output[0]))
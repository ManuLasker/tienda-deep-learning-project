import torch
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Tuple, Union
from base64 import b64decode, b64encode
from io import BytesIO
from PIL import Image


class ImageUtilities:
    # std and mean configuration
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    @staticmethod
    def rectangular_resize_with_pad(
        img: np.ndarray,
        new_shape: Union[Tuple[int, int], int] = (640, 640),
        color: Union[Tuple[int, int, int], int] = (114, 114, 114),
        auto: bool = True,
        scaleFill: bool = False,
        scaleup: bool = True,
    ):
        """Resize with pad a numpy array image in BGR format to a specified size,
        but this resize is rectangular that means it will keep one dimension;
        but the other will shrink to fit some rectangular ratio.

        Args:
            img (np.ndarray): numpy image in BGR format. [H, W, C]
            new_shape (Union[Tuple[int, int], int], optional): new shape to resize. Defaults to (640, 640).
            color (Union[Tuple[int, int, int], int], optional): RGB color or int color to fill padding with.
                                                                Defaults to (114, 114, 114).
            auto (bool, optional): Get automatic minimum rectangle. Defaults to True.
            scaleFill (bool, optional): Stretch or not. Defaults to False.
            scaleup (bool, optional): scale up or only scale down. Defaults to True.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # validate new_shape and color types
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        if isinstance(color, int):
            color = (color, color, color)

        # scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up
            r = min(r, 1.0)
        # compute padding
        ratio = r, r  # width, height ratios
        new_unpad = (
            int(round(shape[1] * r)),
            int(round(shape[0] * r)),
        )  # width, height
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch image
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding by 2
        dh /= 2  # divide

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img

    @staticmethod
    def _base64_to_pil(base64_image: str) -> Image.Image:
        """Convert base64 image string to PIL image
        Args:
            base64Image (str): base 64 str image decode
        Returns:
            Image.Image: Pil image
        """
        # Select just the image information if there is more information
        if len(base64_image.split(",")) > 1:
            _, base64_image = base64_image.split(",")
        pil_image = Image.open(BytesIO(b64decode(base64_image)))
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        return pil_image

    @staticmethod
    def _pil_to_base64(pil_image: Image.Image) -> str:
        """Convert pil image to base64 encode string format
        Args:
            pil_image (Image.Image): pil Image
        Returns:
            str: string base64 image
        """
        _buffer = BytesIO()
        if pil_image.mode != "RGBA":
            pil_image.save(_buffer, format="JPEG")
        else:
            pil_image.save(_buffer, format="PNG")
        img_str = b64encode(_buffer.getvalue()).decode("utf-8")
        return img_str

    @staticmethod
    def _pil_to_numpy(image: Image.Image, _format: str = "RGB") -> np.ndarray:
        """Convert image to numpy array image of specific format from oposite format.
        from RGB2BGR or BGR2RGB

        Args:
            image (Image.Image): Pil image or tensor image.
            format (str): image format. ("RGB" or "BGR"). Default "RGB"
        Returns:
            np.ndarray: numpy image array.
        """
        image = np.array(image)
        return ImageUtilities._transform_image(image, _format)

    @staticmethod
    def _transform_image(image: np.ndarray, _format: str = "RGB") -> np.ndarray:
        """Convert image numpy array from BGR to RGB and RGB to BGR.

        Args:
            image (np.ndarray): image numpy array
            _format (str, optional): format to transforms. Defaults to "RGB".

        Returns:
            np.ndarray: numpy array with new format.
        """
        if _format == "RGB":
            return cv2.cvtColor(image, code=cv2.COLOR_RGB2BGR)
        elif _format == "BGR":
            return cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_image_from_file(file_path: str) -> np.ndarray:
        """load image from file in BGR format.

        Args:
            file_path (str): /path/to/image string.

        Returns:
            np.ndarray: numpy image array in BGR format.
        """
        return cv2.imread(file_path)

    @staticmethod
    def _to_tensor(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert pil Image or numpy image to tensor Image, if numpy image is np.uint8
        in scale 0 - 255, it transforms to torch.float32 in scale of 0 - 1
        Args:
            image (Union[Image.Image, np.ndarray]): pil or numpy image we want to convert to tensor Image.
        Returns:
            torch.Tensor: tensor image.
        """
        return T.ToTensor()(image)

    @staticmethod
    def _to_pil(image: Union[torch.Tensor, np.ndarray]) -> Image.Image:
        """Convert tensor or numpy Image to PIL image format, numpy image needs to be
        inf np.uint8 format in scale of 0 - 255
        Args:
            image (torch.Tensor): tensor image we want to convert to PIL.
        Returns:
            Image.Image: PIL image format
        """
        if isinstance(image, torch.Tensor):
            return T.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)

    @staticmethod
    def process_fastai_model(
        image: Union[np.ndarray, Image.Image], new_shape: Tuple[int, int]
    ) -> torch.Tensor:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(size=new_shape),
                T.Normalize(mean=ImageUtilities.mean, std=ImageUtilities.std),
            ]
        )
        image: torch.Tensor = transform(image)
        return image.float()

    @staticmethod
    def _base64_to_pil(base64_image: str) -> Image.Image:
        """Convert base64 encoded string image to PIL Image.

        Args:
            base64_image (str): base 64 str image encode

        Returns:
            Image.Image: Pil image
        """
        # Select just the image information if ther is more than one
        if len(base64_image.split(",")) > 1:
            _, base64_image = base64_image.split(",")
        pil_image = Image.open(BytesIO(b64decode(base64_image)))
        if pil_image.mode == "RGBA":
            pil_image.convert("RGB")
        return pil_image
    
    def _bytes_to_pil(bytes_image: bytes) -> Image.Image:
        """Convert bytes image into pil image format.

        Args:
            bytes_image (bytes): bytes image

        Returns:
            Image.Image: pil image.
        """
        pil_image = Image.open(BytesIO(bytes_image))
        if pil_image.mode == "RGBA":
            pil_image.convert("RGB")
        return pil_image

    @staticmethod
    def _load_image_from_base64_image(base64_image: str) -> np.ndarray:
        """load image from  base64 encoded string image to numpy image in BGR format.

        Args:
            base64_image (str): base 64 str image encode

        Returns:
            np.ndarray: numpy image array in BGR format.
        """
        return ImageUtilities._pil_to_numpy(
            ImageUtilities._base64_to_pil(base64_image), _format="BGR"
        )
        
    @staticmethod
    def _load_image_from_bytes_image(bytes_image: bytes) -> np.ndarray:
        """Load image from bytes image object to numpy image in BGR format.

        Args:
            bytes_image (bytes): bytes image object.

        Returns:
            np.ndarray: numpy image array in BGR format.
        """
        pil_image = ImageUtilities._bytes_to_pil(bytes_image)
        pil_image.save("/opt/ml/model/image.jpg")
        return ImageUtilities._pil_to_numpy(
            pil_image, _format="BGR"
        )
import math
from typing import Iterable, Dict, Self, List, Tuple, Any

import numpy as np
import cv2
import shapely
from numpy.typing import NDArray

from src.utils import DatasetType


class BoundingBox:
    """
    Author:
        masoud Masoumi Moghadam: (masouduut94)
    Utility module which gets a numpy array of four items as input and
    can provide a variety of tools related to bounding boxes.
    """
    __slots__ = ['box', 'x1', 'x2', 'y1', 'y2', 'attributes', 'label', 'conf', "annot_id", 'name']

    def __init__(self, x: List | Tuple | NDArray, name: str = None, conf: float = 0.0, label: int = None):
        if isinstance(x, list):
            self.box = [int(i) for i in x]
        elif isinstance(x, np.ndarray):
            self.box = x.astype(int).tolist()
        elif isinstance(x, tuple):
            self.box = [int(i) for i in x]
        else:
            raise ValueError("input x must be a list, tuple or numpy array with four values.")
        self.x1: int = self.box[0]
        self.y1: int = self.box[1]
        self.x2: int = self.box[2]
        self.y2: int = self.box[3]

        self.attributes: Dict[str, Any] = {}
        self.label: int = label
        self.conf: float = conf
        self.annot_id: int | None = None
        self.name: str = name

    def extend_image(self, width_start: int, height_start: int) -> Self:
        self.x1 += width_start
        self.x2 += width_start
        self.y1 += height_start
        self.y2 += height_start
        return self.create([self.x1, self.y1, self.x2, self.y2], name=self.name, conf=self.conf, label=self.label)

    @property
    def width(self) -> int:
        return abs(self.x1 - self.x2)

    @property
    def height(self) -> int:
        return abs(self.y1 - self.y2)

    @property
    def min_x(self) -> int:
        return min([self.x1, self.x2])

    @property
    def max_x(self) -> int:
        return max([self.x1, self.x2])

    @property
    def min_y(self) -> int:
        return min([self.y1, self.y2])

    @property
    def max_y(self) -> int:
        return max([self.y1, self.y2])

    def add_attribute(self, key: str, attribute: Any):
        self.attributes[key] = attribute

    def set_annot_id(self, annot_id: int):
        self.annot_id = annot_id

    def to_albumentations(self) -> List[int | None]:
        return [self.x1, self.y1, self.width, self.height, self.label]

    def to_polygon(self) -> shapely.Polygon:
        polygon = shapely.Polygon(
            shell=[
                (self.x1, self.y1),
                (self.x2, self.y1),
                (self.x2, self.y2),
                (self.x1, self.y2)
            ]
        )
        return polygon

    @property
    def detected(self) -> bool:
        return True if all((self.x1 < self.x2, self.y1 < self.y2)) else False

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.x2, self.y1, self.y2

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.width, self.height

    def to_numpy(self) -> NDArray:
        return np.array([self.x1, self.y1, self.x2, self.y2, self.conf])

    def to_yolo(self, img_width, img_height, seg_type=False, current_type=DatasetType.YoloDatasetType) -> str:
        label = self.label if current_type == DatasetType.YoloDatasetType else self.label - 1
        if seg_type:
            img_dimensions = np.array([img_width, img_height])
            tl = (np.array(self.top_left) / img_dimensions).tolist()
            dl = (np.array(self.down_left) / img_dimensions).tolist()
            dr = (np.array(self.down_right) / img_dimensions).tolist()
            tr = (np.array(self.top_right) / img_dimensions).tolist()
            return f"{label} {tl[0]} {tl[1]} {dl[0]} {dl[1]} {dr[0]} {dr[1]} {tr[0]} {tr[1]}"
        else:
            x_cen = self.x1 + (self.width / 2)
            y_cen = self.y1 + (self.height / 2)
            # x_cen, y_cen = self.center
            x_cen = x_cen / img_width
            y_cen = y_cen / img_height
            bbox_width = self.width / img_width
            bbox_height = self.height / img_height
            return f"{label} {x_cen} {y_cen} {bbox_width} {bbox_height}"

    def to_coco(self, image_id: int) -> Dict[str, int | list]:
        return {
            'iscrowd': 0,
            "bbox_mode": 0,
            "area": self.area,
            'id': self.annot_id,
            'image_id': image_id,
            'category_id': self.label,
            'bbox': [self.x1, self.y1, self.width, self.height],
            'segmentation': [],
        }

    @classmethod
    def from_string(cls, cxywh_string: str, img_width: int, img_height: int) -> Self:
        """
        Create bounding box class from yolo-typed string
        Args:
            cxywh_string:
            img_width:
            img_height:

        Returns:

        """
        line = cxywh_string.rstrip('\n').split(' ')
        c, x_cen, y_cen, bbox_width, bbox_height = [float(i) for i in line]
        x1 = x_cen * img_width - (bbox_width * img_width) // 2
        y1 = y_cen * img_height - (bbox_height * img_height) // 2
        x2 = x_cen * img_width + (bbox_width * img_width) // 2
        y2 = y_cen * img_height + (bbox_height * img_height) // 2
        c = int(c)
        return cls.create(x=[x1, y1, x2, y2], name=f"label_{c}", conf=100, label=c)

    @classmethod
    def from_numpy(cls, np_xyxyc: NDArray) -> Self:
        xyxyc = np_xyxyc.tolist()
        x1, y1, x2, y2, conf = xyxyc
        new_bbox = cls.create(x=[x1, y1, x2, y2], name=None, conf=conf, label=None)
        return new_bbox

    def __repr__(self):
        if self.detected:
            return f"""name={self.name} | center={self.center} | box={self.box}"""
        else:
            return f"""name={self.name} | NOT detected!"""

    @classmethod
    def create(cls, x: list | NDArray | tuple, name: str = None, conf: float = 0.0, label: int = None) -> Self:
        """

        Args:
            label:
            conf:
            name:
            x: list of 4 items indicating [x1, y1, x2, y2]

        Returns:
            instantiate a BoundingBox module in place.
        """
        return cls(x, name=name, conf=conf, label=label)

    @property
    def area(self) -> int:
        """
        Calculates the surface area. useful for IOU!
        """
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)

    @property
    def center(self) -> Tuple[float, float]:
        """
        Attribute indicating the center of bbox
        Returns:

        """
        center_x = self.x1 + int(self.width / 2)
        center_y = self.y1 + int(self.height / 2)
        return center_x, center_y

    def distance(self, coordination: NDArray) -> float:
        """
        Calculate distance between its center to given (x, y)
        References:
            https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
        Args:

        Returns:
            the distance between bounding box and the given coordination
        """

        return float(np.round(np.linalg.norm(np.array(self.center) - coordination), 3))

    def intersection(self, bbox: Self) -> int:
        if isinstance(bbox, list):
            bbox = BoundingBox(bbox)
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        return intersection

    def iou(self, box: list | tuple) -> float:
        """
        Calculates the intersection over union with bbox given
        References:
            https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        Args:
            box: (iterable): it's a tuple/list/numpy array of 4 items x1, y1, x2, y2

        Returns:

        """
        bbox = self.create(box)
        intersection = self.intersection(bbox)

        iou = intersection / float(self.area + bbox.area - intersection)
        # return the intersection over union value
        return iou

    def frame_crop(self, frame: NDArray, margin: int = None) -> NDArray:
        """
        Crop a portion of the image
        Args:
            frame:
            margin:

        Returns:

        """
        h, w, _ = frame.shape
        if margin is not None:
            y1 = (self.y1 - margin) if (self.y1 - margin) > 0 else 0
            y2 = (self.y2 + margin) if (self.y2 + margin) < h else h
            x1 = (self.x1 - margin) if (self.x1 - margin) > 0 else 0
            x2 = (self.x2 + margin) if (self.x2 + margin) < w else w
            f = frame[y1: y2, x1: x2, :]
        else:
            f = frame[self.y1: self.y2, self.x1: self.x2, :]

        h, w, _ = f.shape
        pixels = abs(w - h)

        if w > h:
            f = cv2.copyMakeBorder(
                f, top=pixels // 2, bottom=pixels // 2,
                left=0, right=0,
                borderType=cv2.BORDER_CONSTANT
            )
        else:
            f = cv2.copyMakeBorder(
                f, top=0, bottom=0,
                left=pixels // 2, right=pixels // 2,
                borderType=cv2.BORDER_CONSTANT
            )
        return f

    @property
    def top_left(self) -> Tuple[int, int]:
        return self.min_x, self.min_y

    @property
    def down_left(self) -> Tuple[int, int]:
        return self.min_x, self.max_y

    @property
    def top_right(self) -> Tuple[int, int]:
        return self.max_x, self.min_y

    @property
    def down_right(self) -> Tuple[int, int]:
        return self.max_x, self.max_y

    @property
    def down_center(self) -> Tuple[int, int]:
        x_center = (self.down_left[0] + self.down_right[0]) // 2
        y_center = (self.down_left[1] + self.down_right[1]) // 2
        return x_center, y_center

    @property
    def top_center(self) -> Tuple[int, int]:
        x_center = (self.top_left[0] + self.top_right[0]) // 2
        y_center = (self.top_left[1] + self.top_right[1]) // 2
        return x_center, y_center

    @property
    def diameter(self) -> float:
        return math.sqrt(self.width ** 2 + self.height ** 2)

import math
from typing import Iterable, Dict

import numpy as np
import cv2
import shapely
from numpy.typing import NDArray

from src.utils import DatasetType


class BoundingBox:
    """
    Author:
        masoud Masoumi Moghadam: (masouduut94)
    Utility module which gets a numpy array of 4 items as input and
    can provide variety of tools related to bounding boxes.
    """

    def __init__(self, x, name=None, conf=0.0, label: int = None):
        if isinstance(x, list):
            self.box = [int(i) for i in x]
        elif isinstance(x, dict):
            x1 = x['x1']
            x2 = x['x2']
            y1 = x['y1']
            y2 = x['y2']
            self.box = [int(i) for i in [x1, y1, x2, y2]]
        elif isinstance(x, np.ndarray):
            self.box = x.astype(int).tolist()
        self.x1, self.y1, self.x2, self.y2 = self.box

        self.attributes = {}
        self.label = label
        self.conf = conf
        self.annot_id = None
        self.name = name
        self.random_color = tuple(np.random.randint(low=0, high=254, size=(3,)).tolist())

    def extend_image(self, width_start, height_start):
        self.x1 += width_start
        self.x2 += width_start
        self.y1 += height_start
        self.y2 += height_start
        return self.create([self.x1, self.y1, self.x2, self.y2], name=self.name, conf=self.conf, label=self.label)

    @property
    def width(self):
        return abs(self.x1 - self.x2)

    @property
    def height(self):
        return abs(self.y1 - self.y2)

    @property
    def min_x(self):
        return min([self.x1, self.x2])

    @property
    def max_x(self):
        return max([self.x1, self.x2])

    @property
    def min_y(self):
        return min([self.y1, self.y2])

    @property
    def max_y(self):
        return max([self.y1, self.y2])

    def add_attribute(self, key, attribute):
        self.attributes[key] = attribute

    def set_annot_id(self, annot_id):
        self.annot_id = annot_id

    def to_albumentations(self):
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
    def detected(self):
        return True if all((self.x1 < self.x2, self.y1 < self.y2)) else False

    def to_xyxy(self):
        return self.x1, self.x2, self.y1, self.y2

    def to_xywh(self):
        return self.x1, self.y1, self.width, self.height

    def to_numpy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2, self.conf])

    def to_yolo(self, img_width, img_height, seg_type=False, current_type=DatasetType.YoloDatasetType):
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

    def to_coco(self, image_id: int) -> Dict:
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

    def to_coco_eval(self, image_id: int) -> Dict:
        return {
            "bbox": [self.x1, self.y1, self.width, self.height],
            "category_id": self.label,
            "image_id": image_id,
            "score": self.conf,
        }

    @classmethod
    def from_string(cls, cxywh_string: str, img_width: int, img_height: int):
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
    def from_numpy(cls, np_xyxyc: NDArray):
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
    def create(cls, x, name=None, conf=0.0, label: int = None):
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
    def area(self):
        """
        Calculates the surface area. useful for IOU!
        """
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)

    @property
    def center(self):
        """
        Attribute indicating the center of bbox
        Returns:

        """
        center_x = self.x1 + int(self.width / 2)
        center_y = self.y1 + int(self.height / 2)
        return center_x, center_y

    def distance(self, coordination: np.ndarray):
        """
        Calculate distance between its center to given (x, y)
        References:
            https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
        Args:

        Returns:
            the distance between bounding box and the given coordination
        """

        return np.round(np.linalg.norm(np.array(self.center) - coordination), 3)

    def intersection(self, bbox):
        if isinstance(bbox, list):
            bbox = BoundingBox(bbox)
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        return intersection

    def iou(self, box: list | tuple):
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

    def iou_overlap(self, bbox: list | tuple):
        box1 = np.array([self.x1, self.y1, self.x2, self.y2])
        box2 = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        top_left = np.maximum(box1[:2], box2[:2])
        bottom_right = np.minimum(box1[2:], box2[2:])
        b = BoundingBox([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
        return self.iou(b.box)

    def frame_crop(self, frame, margin=None):
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
    def top_left(self):
        return self.min_x, self.min_y

    @property
    def down_left(self):
        return self.min_x, self.max_y

    @property
    def top_right(self):
        return self.max_x, self.min_y

    @property
    def down_right(self):
        return self.max_x, self.max_y

    @property
    def down_center(self):
        x_center = (self.down_left[0] + self.down_right[0]) // 2
        y_center = (self.down_left[1] + self.down_right[1]) // 2
        return x_center, y_center

    @property
    def top_center(self):
        x_center = (self.top_left[0] + self.top_right[0]) // 2
        y_center = (self.top_left[1] + self.top_right[1]) // 2
        return x_center, y_center

    @property
    def diameter(self):
        return math.sqrt(self.width ** 2 + self.height ** 2)

from typing import Tuple, Dict

import cv2
from pathlib import Path

from numpy.typing import NDArray
from typing_extensions import List
from src.utils import BoundingBox, DatasetType


class ImageAnnotations(object):
    """
    This module provides utility functions over image and its annotations.
    It can output json to COCO format and darknet.
    Capabilities:
        - It can filter out the categories of objects in the output json.
        - It can help you plot and test the objects in the image to check
             out the annotations. you can also filter certain objects in
             plot if you want
    """

    def __init__(self, image_id: int | None, file_name: Path):
        self.image_id: int = image_id
        self.file_name: Path = file_name
        image: NDArray = self.get_image()
        self.height, self.width, _ = image.shape
        self.annotations: List[BoundingBox] = []
        del image

    def get_image(self) -> NDArray:
        return cv2.imread(self.file_name.as_posix())

    def add_annotation(self, bbox: BoundingBox):
        self.annotations.append(bbox)

    def set_annotations(self, bounding_boxes: List[BoundingBox]):
        self.annotations = bounding_boxes

    def extend_annotations(self, bounding_boxes: List[BoundingBox]):
        self.annotations.extend(bounding_boxes)

    def __len__(self):
        return len(self.annotations)

    def __repr__(self):
        return f"""(ImgAnnot: file_name={self.file_name}, image_id={self.image_id}, n_annotations={len(self)})"""

    def to_coco_fmt(self, update_labels: bool = False) -> tuple:
        """
        Prepares the image details and the annotations on an image.
        Notes:
            The annotations must be assigned a unique integer. We have to
            do that only when we have access to all the annotations of all
            images. So there is a need for loop over all images and their
            annotations to assign unique integer for both image and
            corresponding annotations.
        Returns:

        """

        annotations_info = []
        for annot in self.annotations:
            coco_info = annot.to_coco(image_id=self.image_id)
            cat_id = annot.label if not update_labels else annot.label + 1
            coco_info['category_id'] = cat_id
            coco_info['attributes'] = {}
            coco_info['attributes']["conf"] = annot.conf
            for attr_key, attr_value in annot.attributes.items():
                coco_info['attributes'][attr_key] = attr_value
            annotations_info.append(coco_info)

        image_info = {
            'id': self.image_id,
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name.name
        }
        return image_info, annotations_info

    def to_yolo_fmt(self, current_type: int = DatasetType.YoloDatasetType) -> str:
        txt = ""
        for i, annot in enumerate(self.annotations):
            t = annot.to_yolo(self.width, self.height, current_type=current_type)
            if i != len(self.annotations) - 1:
                txt = txt + t + '\n'
            else:
                txt += t
        return txt

    def coco_plot(self, image_id: int):
        image = cv2.imread(self.file_name.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colormap = (255, 200, 0)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 3

        for annot in self.annotations:
            annot_info = annot.to_coco(image_id=image_id)
            category_id = annot_info['category_id']
            x1, y1, x2, y2 = annot_info['bbox']
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=colormap, thickness=3)
            text_org = (x1 - 5, y1 - 5)
            image = cv2.putText(
                image, f"class {category_id}", org=text_org, fontFace=font_face, fontScale=font_scale,
                color=colormap, thickness=thickness
            )
        return image

    @staticmethod
    def get_coco_categories(category_map: List[str]):
        categories = [{'supercategory': item, 'id': i + 1, 'name': item} for i, item in enumerate(category_map)]
        return categories

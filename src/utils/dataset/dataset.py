import os
import cv2
import json
import yaml
import shutil

import numpy as np
import oyaml as yml
import pandas as pd
from tqdm import tqdm
import itertools as it
from pathlib import Path
import albumentations as A
from natsort import natsorted
from pycocotools.coco import COCO
from rich.progress import track
from typing_extensions import List, Dict, Tuple
from albumentations.core.transforms_interface import DualTransform
from .image_annotation import ImageAnnotations

from src.utils import DatasetType, BoxPlotType, BoundingBox, sv_plot


# from src.utils.general import save_img_slice_and_labels


class DatasetManager:
    """
        Be able to initiate data from
            1. Yolo Directory/yaml file ?
            2. COCO json
            3. ImageAnnot annotations

        Be able to output:
            1. COCO json and images
            2. Augment data
            3. CVAT output.
            4. Yolo data
            5. Create yaml file based on clusters.
            6. Create json file for COCO dataset.


        Other utilities:
            1. Handle images
            2. Convert to different types
            3. Slice images and bounding boxes.
            4. Handle segmentation/detection
            5. Generate masks
            6. Slice images/masks too.
            7. Augment images with different techniques
            8. Plot images/bboxes/masks.
            9. Use multithreading (joblib)
            10. Add a function that triggers the image/annotation loading (use it
                only when it is required to convert datasets to each other, and nothing else...
            11. Before image/annots loading, it can produce the plots.
    """

    def __init__(self):
        self.image_annots: List[ImageAnnotations] = []
        self.dataset_type: int = DatasetType.NoValue
        self.categories: List[Dict] = []
        self.annot_id = 1
        self.image_id = 1

    @staticmethod
    def create_image_annot(image: str | Path, bboxes: List[BoundingBox]):
        img_path = Path(image) if isinstance(image, str) else image
        img_annot = ImageAnnotations(image_id=None, file_name=img_path)
        for bbox in bboxes:
            img_annot.add_annotation(bbox)
        return img_annot

    def reset_image_annots(self):
        self.image_annots.clear()
        self.annot_id = 1
        self.image_id = 1

    @staticmethod
    def get_yolo_by_name(image_path, yaml_file) -> ImageAnnotations | None:
        yaml_data = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        names = yaml_data['names']

        image_path = Path(image_path)
        label_path = image_path.parent.parent.parent / f'labels/train/{image_path.stem}.txt'
        if not label_path.is_file():
            print(f"The label doesn't exist {label_path.as_posix()}")
            return None

        img_annot = ImageAnnotations(image_id=None, file_name=image_path)
        lines = open(label_path.as_posix()).readlines()
        for line in lines:
            bbox = BoundingBox.from_string(line, img_width=img_annot.width, img_height=img_annot.height)
            bbox.name = names[bbox.label]
            img_annot.add_annotation(bbox)
        return img_annot

    @staticmethod
    def get_coco_by_name(image_path, coco_json) -> ImageAnnotations | None:
        coco = COCO(coco_json)
        image_path = Path(image_path)
        request_img_id = [k for k, v in coco.imgs.items() if v['file_name'] == image_path.name]
        if not len(request_img_id):
            return None
        img_id = request_img_id[0]
        cat2name = {c['id']: c['name'] for i, c in coco.cats.items()}
        img_annot = ImageAnnotations(image_id=None, file_name=image_path)
        annot_ids = coco.getAnnIds(imgIds=img_id)
        for annot_id in annot_ids:
            x1, y1, w, h = coco.anns[annot_id]['bbox']
            cat_id = coco.anns[annot_id]['category_id']
            bbox = BoundingBox(x=[x1, y1, x1 + w, y1 + h], name=cat2name[cat_id], label=cat_id - 1)
            img_annot.add_annotation(bbox)
        return img_annot

    def from_yolo_path(self, yaml_file: str, yolo_data_path: str):
        self.dataset_type = DatasetType.YoloDatasetType
        yaml_data = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        names = yaml_data['names']
        categories = []

        for label, name in names.items():
            categories.append({'id': label, "name": name})
        images_path = Path(yolo_data_path) / 'images'
        labels_path = Path(yolo_data_path) / 'labels'
        assert images_path.is_dir(), f'image path "{images_path.as_posix()}" does not exist '
        assert labels_path.is_dir(), f'label path "{labels_path.as_posix()}" does not exist '

        images = sorted([img for img in images_path.rglob('*.jpg')])
        labels = sorted([label for label in labels_path.rglob('*.txt')])
        pairs = list(zip(images, labels))

        annot_id = 0
        for img_id, (image_path, label_path) in enumerate(tqdm(pairs)):
            img_annot = ImageAnnotations(image_id=img_id, file_name=image_path)
            lines = open(label_path).readlines()
            for line in lines:
                bbox = BoundingBox.from_string(line, img_width=img_annot.width, img_height=img_annot.height)
                bbox.set_annot_id(annot_id)
                bbox.name = names[bbox.label]
                img_annot.add_annotation(bbox)
            self.image_annots.append(img_annot)
        self.categories = categories

    def from_coco_json(self, coco_json: str, coco_image_path: str, K=None):
        self.dataset_type = DatasetType.COCODatasetType
        coco = COCO(coco_json)
        image_annots = []
        self.categories = [{"id": cat_info['id'], "name": cat_info['name']} for id_, cat_info in coco.cats.items()]

        cat2name = {c['id']: c['name'] for c in self.categories}
        for k, img in tqdm(coco.imgs.items(), desc='creating image_annotations'):
            img_annot = ImageAnnotations(image_id=img['id'], file_name=Path(coco_image_path) / img['file_name'])
            annot_ids = coco.getAnnIds(imgIds=self.image_id)
            self.image_id += 1
            for annot_id in annot_ids:
                x1, y1, w, h = coco.anns[annot_id]['bbox']
                cat_id = coco.anns[annot_id]['category_id']
                bbox = BoundingBox(x=[x1, y1, x1 + w, y1 + h], name=cat2name[cat_id], label=cat_id)
                bbox.annot_id = self.annot_id
                self.annot_id += 1
                img_annot.add_annotation(bbox)
            image_annots.append(img_annot)
            if K is not None:
                if k >= K:
                    break
        self.image_annots.extend(image_annots)

    def from_image_annots(self, image_annots: List[ImageAnnotations], categories: List[dict],
                          data_type: int = DatasetType.COCODatasetType):
        self.dataset_type = data_type
        for img_id, img_annot in enumerate(image_annots):
            image_annots[img_id].image_id = self.image_id
            self.image_id += 1
            for index, _ in enumerate(img_annot.annotations):
                image_annots[img_id].annotations[index].annot_id = self.annot_id
                self.annot_id += 1
        self.image_annots.extend(image_annots)
        self.categories = categories

    def set_categories(self, categories: dict):
        self.categories = categories

    def to_yaml(self):
        outputs = {'names': []}
        for item in self.categories:
            id_ = item['id']
            category = item['name']
            outputs['names'][id_] = category
        return outputs

    def to_coco(self, output_path: str, to_eval=False):
        images = []
        annotations = []
        img_path = Path(output_path)
        img_path.mkdir(exist_ok=True)
        if to_eval:
            for i, img_annot in enumerate(track(self.image_annots, description='Export to coco eval: ')):
                if self.dataset_type == DatasetType.YoloDatasetType:
                    _, annotations_info = img_annot.to_coco_fmt(update_labels=True, coco_eval=True)
                elif self.dataset_type == DatasetType.COCODatasetType:
                    _, annotations_info = img_annot.to_coco_fmt(update_labels=False, coco_eval=True)
                else:
                    raise ValueError("self.dataset_type can be either the type of yolo or coco.")
                annotations.extend(annotations_info)

            output = annotations.copy()
            output.sort(key=lambda x: x['score'])
        else:
            for i, img_annot in enumerate(track(self.image_annots, description='Export to coco: ')):
                if not os.path.isfile(img_path / img_annot.file_name.name):
                    shutil.copy2(img_annot.file_name.as_posix(), img_path / img_annot.file_name.name)
                new_img_annot = ImageAnnotations(image_id=img_annot.image_id, file_name=img_path / img_annot.file_name.name)
                new_img_annot.annotations = img_annot.annotations
                if self.dataset_type == DatasetType.YoloDatasetType:
                    img_info, annot_info = new_img_annot.to_coco_fmt(update_labels=True)
                elif self.dataset_type == DatasetType.COCODatasetType:
                    img_info, annot_info = new_img_annot.to_coco_fmt(update_labels=False)
                else:
                    raise ValueError("self.dataset_type can be either the type of yolo or coco.")
                images.append(img_info)
                annotations.extend(annot_info)

            if self.dataset_type == DatasetType.YoloDatasetType:
                att = [
                    {
                        "name": "text",
                        "mutable": True,
                        "input_type": "text",
                        "values": [
                            ""
                        ]
                    },
                    {
                        "name": "conf",
                        "mutable": True,
                        "input_type": "number",
                        "values": [
                            "0;100;.01"
                        ]
                    }
                ]
                categories = [{'id': cat['id'] + 1, 'name': cat['name'], "attributes": att} for cat in self.categories]

            else:
                categories = self.categories

            output = {
                'annotations': annotations,
                'images': images,
                'categories': categories,
                'info': {},
                'licenses': []
            }
        output_file = os.path.join(output_path, 'annotations.json')
        with open(output_file, 'w') as file:
            json.dump(output, file, sort_keys=True, indent=2)
        return output_file

    def to_yolo(self, output_path: str, train_ratio: float = 0.9):
        output_path = Path(output_path)
        train_size = int(len(self.image_annots) * train_ratio) if train_ratio else len(self.image_annots)
        bar = track(self.image_annots, description='preparing yolo data')
        for i, img_annot in enumerate(bar):
            directory = 'train' if i < train_size else 'val'
            text = img_annot.to_yolo_fmt(seg_type=False, current_type=self.dataset_type)
            stem = img_annot.file_name.stem
            image_folder_path = output_path / f'images/{directory}'
            label_path = output_path / f'labels/{directory}'
            os.makedirs(image_folder_path, exist_ok=True)
            os.makedirs(label_path, exist_ok=True)
            lbl_name = label_path / (stem + '.txt')
            with open(lbl_name, 'w') as file:
                file.write(text)
            shutil.copy2(img_annot.file_name, image_folder_path)

    def augment(self, image_annots: List[ImageAnnotations], temp_path: str, rot90: bool = False, rot180: bool = False,
                rot270: bool = False, flip_lr: bool = False, flip_ud: bool = False):
        new_img_annots: List[ImageAnnotations] = []
        temp_path = Path(temp_path)
        if flip_lr:
            augment_technique = A.HorizontalFlip(p=1)
            output = self.transform(image_annots, temp_path, augment_technique, "_lr_flip")
            new_img_annots.extend(output)
        if flip_ud:
            augment_technique = A.VerticalFlip(p=1)
            output = self.transform(image_annots, temp_path, augment_technique, "_ud_flip")
            new_img_annots.extend(output)
        if rot90:
            augment_technique = A.Affine(rotate=[90, 90], p=1, mode=cv2.BORDER_CONSTANT, fit_output=True)
            output = self.transform(image_annots, temp_path, augment_technique, "_rot90")
            new_img_annots.extend(output)
        if rot180:
            augment_technique = A.Affine(rotate=[180, 180], p=1, mode=cv2.BORDER_CONSTANT, fit_output=True)
            output = self.transform(image_annots, temp_path, augment_technique, "_rot180")
            new_img_annots.extend(output)
        if rot270:
            augment_technique = A.Affine(rotate=[270, 270], p=1, mode=cv2.BORDER_CONSTANT, fit_output=True)
            output = self.transform(image_annots, temp_path, augment_technique, "_rot270")
            new_img_annots.extend(output)

        for i, img_annot in enumerate(new_img_annots):
            new_img_annots[i].image_id = self.image_id

            self.image_id += 1
            for ii, bbox in enumerate(img_annot.annotations):
                new_img_annots[i].annotations[ii].annot_id = self.annot_id
                self.annot_id += 1
        self.image_annots.extend(new_img_annots)

    def transform(self, image_annots: List[ImageAnnotations], temp_path: Path, augment_technique: DualTransform,
                  output_suffix: str) -> List[ImageAnnotations]:
        cat2name = {c['id']: c['name'] for c in self.categories}
        results = []
        for img_annot in tqdm(image_annots, desc=f"{output_suffix}"):
            bboxes: list = [box.to_albumentations() for box in img_annot.annotations]
            image = img_annot.get_image()
            transform = A.Compose([
                augment_technique,
            ], bbox_params=A.BboxParams(format='coco'))
            transformed = transform(image=image, bboxes=bboxes)
            tr_image, tr_bboxes = transformed['image'], transformed['bboxes']
            name = temp_path / (img_annot.file_name.stem + f'{output_suffix}.jpg')
            cv2.imwrite(name.as_posix(), tr_image)
            new_img_annot = ImageAnnotations(image_id=None, file_name=name)

            for i, (tr_bbox, bbox) in enumerate(list(zip(tr_bboxes, bboxes))):
                x1, y1, w, h, label = [int(i) for i in tr_bbox]
                b = BoundingBox(x=[x1, y1, x1 + w, y1 + h], label=label, name=cat2name[label])
                new_img_annot.add_annotation(b)
            results.append(new_img_annot)
        return results

    def replace_annot_label(self, substitution_dict: dict) -> None:
        for i, img_annot in enumerate(tqdm(self.image_annots, desc='replacing old ids')):
            for ii, annot in enumerate(img_annot.annotations):
                self.image_annots[i].annotations[ii].label = substitution_dict[annot.label]

    @staticmethod
    def plot_image_annot(image_annot: ImageAnnotations, plot_type=BoxPlotType.Color) -> np.ndarray:
        image = image_annot.get_image()
        bboxes = image_annot.annotations
        plot = sv_plot(image, bboxes, draw_label=True, draw_name=True, plot_type=plot_type)
        return plot

    def to_pdf(self, output_path: str):
        from pypdf import PdfReader, PdfWriter
        from pypdf.annotations import FreeText
        from PIL import Image

        writer = PdfWriter()

        # Fill the writer with the pages you want
        pdf_path = os.path.join(output_path, "pdf_hotspots.pdf")

        img_path = Path(output_path)
        img_path.mkdir(exist_ok=True)

        for i, img_annot in enumerate(track(self.image_annots, description='Export to PDF: ')):
            image = Image.open(img_annot.file_name)
            image_pdf_path = os.path.join(output_path, img_annot.file_name.stem + ".pdf")
            image.save(image_pdf_path)

            reader = PdfReader(image_pdf_path)
            page = reader.pages[0]
            writer.add_page(page)

            # writer.add_blank_page(width=img_annot.width, height=img_annot.height)
            # data = img_annot.file_name.stem
            # writer.add_attachment(img_annot.file_name.name, data)

            for annot in img_annot.annotations:
                if annot.name == "text":
                    continue

                scale_x = 1
                scale_y = 1
                left = annot.box[0]
                top = annot.box[1]
                width = annot.box[2] - left
                height = annot.box[3] - top

                left = left * scale_x
                top = (img_annot.height - top - height) * scale_y
                right = left + width
                bottom = top + height

                annotation = FreeText(
                    text=f"{annot.name} - {annot.attributes['text']}",
                    rect=(left, top, right, bottom),
                    font="Arial",
                    bold=True,
                    italic=True,
                    font_size="20pt",
                    font_color="00ff00",
                    border_color="0000ff",
                    background_color=None,
                )
                writer.add_annotation(page_number=0, annotation=annotation)

            # Write the annotated file to disk
            with open(pdf_path, "wb") as fp:
                writer.write(fp)
        return pdf_path

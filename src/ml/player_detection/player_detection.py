# from abc import ABC
import cv2
from tqdm import tqdm
from typing import List
from pathlib import Path
from ultralytics import YOLO
from numpy.typing import NDArray

from src.utils import BoundingBox, CourtCoordinates, SuperVisionPlot, BoxPlotType


# weights = 'yolov8n.pt'


class PlayerDetector:
    def __init__(self, cfg, court_dict: dict = None):
        self.name = 'player'
        self.model = YOLO(cfg['weight'])
        self.court = CourtCoordinates(court_dict) if court_dict is not None else None

    def predict(self, inputs: NDArray) -> list[BoundingBox]:
        results = self.model(inputs, classes=0)
        confs = results[0].boxes.conf.cpu().detach().numpy().tolist()
        boxes = results[0].boxes.xyxy.cpu().detach().numpy().tolist()

        detections: List[BoundingBox] = []
        for box, conf in zip(boxes, confs):
            b = BoundingBox(box, name=self.name, conf=float(conf), label=0)
            detections.append(b)
        detections.sort(key=lambda x: (x.conf, x.area), reverse=True)
        return detections

    def batch_predict(self, inputs: List[NDArray]) -> List[List[BoundingBox]]:
        outputs = self.model(inputs, verbose=False, classes=0)
        results = []
        for res in outputs:
            confs = res.boxes.conf.cpu().detach().numpy().tolist()
            boxes = res.boxes.xyxy.cpu().detach().numpy().tolist()

            detections: List[BoundingBox] = []
            for box, conf in zip(boxes, confs):
                b = BoundingBox(box, name=self.name, conf=float(conf), label=0)
                detections.append(b)
            detections.sort(key=lambda x: (x.conf, x.area), reverse=True)
            results.append(detections)
        return results

    def filter(self, bboxes: List[BoundingBox], keep: int = None, by_bbox_size: bool = True,
               by_zone: bool = True):
        """
        filter the bounding boxes of people based on the size of bounding box,
        whether their steps are in the court_segmentation.
        Args:
            by_zone:
            bboxes:
            keep:
            by_bbox_size:

        Returns:

        """
        if self.court is not None:
            # Keep the player_detection that their legs keypoint (x, y) are
            # inside the polygon-shaped court_segmentation ...
            if by_zone:
                bboxes = [b for b in bboxes if
                          any([self.court.is_inside_main_zone(b.down_left),
                               self.court.is_inside_main_zone(b.down_right),
                               self.court.is_inside_main_zone(b.center),
                               self.court.is_inside_front_zone(b.down_left),
                               self.court.is_inside_front_zone(b.down_right)])]
        if by_bbox_size:
            bboxes.sort(key=lambda x: (x.conf, x.area))
        else:
            bboxes.sort(key=lambda x: x.conf)
        # https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python
        return bboxes[:keep] if keep is not None else bboxes

    @staticmethod
    def draw(input_frame: NDArray, items: List[BoundingBox]):
        if not len(items):
            return input_frame

        input_frame = SuperVisionPlot.bbox_plot(input_frame, items, plot_type=BoxPlotType.Corner)
        return input_frame


if __name__ == '__main__':
    video = '/home/masoud/Desktop/projects/volleyball_analytics/data/raw/videos/train/11.mp4'
    output = '/home/masoud/Desktop/projects/volleyball_analytics/runs/detect/onnx'
    cfg = {
        'weight': '/home/masoud/Desktop/projects/yolov8-tensorrt-test/weights/yolov8s.onnx',
        "labels": {0: 'person'}
    }

    player_detector = PlayerDetector(cfg=cfg)
    cap = cv2.VideoCapture(video)
    assert cap.isOpened()

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = Path(video) / (Path(video).stem + 'person_onnx_output.mp4')
    writer = cv2.VideoWriter(output_file.as_posix(), fourcc, fps, (w, h))
    lst = []
    for fno in tqdm(list(range(n_frames))):
        cap.set(1, fno)
        status, frame = cap.read()
        frame = cv2.resize(frame, (640, 640))
        bboxes = player_detector.predict(lst)
        frame = player_detector.draw(frame, bboxes)
        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'saved results in {output_file}')

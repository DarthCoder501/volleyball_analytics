from typing import List
from ultralytics import YOLO
from numpy.typing import NDArray

from src.utils import BoundingBox, CourtCoordinates

# weights = 'yolov8n-seg.pt'
__all__ = ['PlayerSegmentor']

from src.utils import SuperVisionPlot, KeyPointBox, BoxPlotType, KeyPointPlotType


class PlayerSegmentor:
    def __init__(self, cfg, court_dict: dict = None):
        self.name = 'player'
        self.model = YOLO(cfg['weight'])
        self.labels = {i: n for i, n in enumerate(self.model.names)}
        self.court = CourtCoordinates(court_dict) if court_dict is not None else None

    def predict(self, input_frame: NDArray) -> list[BoundingBox]:
        results = self.model(input_frame, verbose=False, classes=0)
        confs = results[0].boxes.conf.cpu().detach().numpy().tolist()
        boxes = results[0].boxes.xyxy.cpu().detach().numpy().tolist()

        detections: List[BoundingBox] = []
        for box, conf in zip(boxes, confs):
            # TODO: make it suitable for multi-class yolo.
            b = BoundingBox(box, name=self.name, conf=float(conf))
            detections.append(b)
        detections.sort(key=lambda x: (x.conf, x.area), reverse=True)
        return detections

    def batch_predict(self, input_frame: List[NDArray]) -> list[List[BoundingBox]]:
        outputs = self.model(input_frame, verbose=False, classes=0)
        results = []
        for output in outputs:
            confs = output.boxes.conf.cpu().detach().numpy().tolist()
            boxes = output.boxes.xyxy.cpu().detach().numpy().tolist()

            detections: List[BoundingBox] = []
            for box, conf in zip(boxes, confs):
                b = BoundingBox(box, name=self.name, conf=float(conf))
                detections.append(b)
            detections.sort(key=lambda x: (x.conf, x.area), reverse=True)
            results.append(detections)
        return results

    def filter(self, bboxes: List[BoundingBox], keep: int = None, by_bbox_size: bool = True, by_zone: bool = True):
        """
        filter the bounding boxes of people based on the size of bounding box,
        and also whether their steps are in the court_segmentation or not.
        Args:
            by_zone:
            bboxes:
            keep: how many people to keep. normally 12 person. (6 for one team, 6 for other team)
            by_bbox_size:

        Returns:

        """
        if self.court is not None:
            # Keep the player_detection that their legs keypoint (x, y)
            # are inside the polygon-shaped court_segmentation ...
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
    def draw(input_frame: NDArray, items: List[BoundingBox | KeyPointBox]):
        if not len(items):
            return input_frame

        if isinstance(items[0], BoundingBox):
            input_frame = SuperVisionPlot.bbox_plot(input_frame, items, plot_type=BoxPlotType.Corner)
        else:
            input_frame = SuperVisionPlot.keypoint_plot(input_frame, items, plot_type=KeyPointPlotType.Vertex)

        return input_frame

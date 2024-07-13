from typing import List
from numpy.typing import NDArray
from ultralytics import YOLO

from src.utils import BoundingBox, KeyPointBox, BoxPlotType, KeyPointPlotType, SuperVisionPlot


class PoseEstimator:
    def __init__(self, cfg: dict):
        self.model = YOLO(cfg['weight'])

    def predict(self, inputs: NDArray) -> List[KeyPointBox]:
        results = self.model(inputs, verbose=False)
        confs = results[0].boxes.conf.cpu().detach().numpy().tolist()
        kps = results[0].key_points.xy.cpu().detach().numpy().astype(int)
        keypoints = []
        for kp, conf in zip(kps, confs):
            kp = KeyPointBox(key_points=kp, conf=conf, name="player")
            keypoints.append(kp)
        return keypoints

    def batch_predict(self, inputs: List[NDArray]) -> List[List[KeyPointBox]]:
        outputs = self.model(inputs, verbose=False)
        results = []
        for res in outputs:
            confs = res.boxes.conf.cpu().detach().numpy().tolist()
            kps = res.key_points.xy.cpu().detach().numpy().astype(int)

            keypoints: List[KeyPointBox] = []
            for kp, conf in zip(kps, confs):
                kp = KeyPointBox(key_points=kp, conf=conf, name="player")
                keypoints.append(kp)
            results.append(keypoints)
        return results

    @staticmethod
    def draw(frame: NDArray, items: List[BoundingBox | KeyPointBox]):
        if not len(items):
            return frame

        if isinstance(items[0], BoundingBox):
            frame = SuperVisionPlot.bbox_plot(frame, items, plot_type=BoxPlotType.Corner)
        else:
            frame = SuperVisionPlot.keypoint_plot(frame, items, plot_type=KeyPointPlotType.Vertex)

        return frame

from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm
from ultralytics import YOLO
from numpy.typing import NDArray
from src.utils import BoundingBox, SuperVisionPlot, BoxPlotType, KeyPointBox, KeyPointPlotType


class BallSegmentor:
    def __init__(self, cfg: dict):
        self.model = YOLO(cfg['weight'])
        self.labels = {i: n for i, n in enumerate(self.model.names)}

    def predict(self, inputs: NDArray) -> List[BoundingBox]:
        outputs = self.model(inputs, verbose=False)
        confs = outputs[0].boxes.conf.cpu().detach().numpy().tolist()
        boxes = outputs[0].boxes.xyxy.cpu().detach().numpy().tolist()

        results: List[BoundingBox] = []
        for box, conf in zip(boxes, confs):
            b = BoundingBox(box, name='ball', conf=float(conf), label=0)
            results.append(b)
        results.sort(key=lambda x: (x.conf, x.area), reverse=True)
        return results

    def batch_predict(self, inputs: List[NDArray]) -> List[List[BoundingBox]]:
        outputs = self.model(inputs, verbose=False)
        results = []
        for res in outputs:
            confs = res.boxes.conf.cpu().detach().numpy().tolist()
            boxes = res.boxes.xyxy.cpu().detach().numpy().tolist()

            detections: List[BoundingBox] = []
            for box, conf in zip(boxes, confs):
                b = BoundingBox(box, name='ball', conf=float(conf), label=0)
                detections.append(b)
            detections.sort(key=lambda x: (x.conf, x.area), reverse=True)
            results.append(detections)
        return results

    @staticmethod
    def draw(frame: NDArray, items: List[BoundingBox]):
        if not len(items):
            return frame

        frame = SuperVisionPlot.bbox_plot(frame, items, plot_type=BoxPlotType.Corner)
        return frame


if __name__ == '__main__':
    video = '/home/masoud/Desktop/projects/volleyball_analytics/data/raw/videos/train/nian1.mp4'
    output = '/home/masoud/Desktop/projects/volleyball_analytics/runs/DEMO'

    cfg = {
        'weight': '/home/masoud/Desktop/projects/volleyball_analytics/runs/detect/train/weights/best.pt',
    }

    ball_detector = BallSegmentor(cfg=cfg)
    cap = cv2.VideoCapture(video)
    assert cap.isOpened()

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = Path(output) / (Path(video).stem + '__NEW_BALL_DET.mp4')
    writer = cv2.VideoWriter(output_file.as_posix(), fourcc, fps, (w, h))

    for fno in tqdm(list(range(n_frames))):
        cap.set(1, fno)
        status, frame = cap.read()
        bboxes = ball_detector.predict(frame)
        frame = ball_detector.draw(frame, bboxes)
        if fno > 1500:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    # cv2.destroyAllWindows()
    print(f'saved results in {output_file}')

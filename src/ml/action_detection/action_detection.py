from typing import List
from ultralytics import YOLO
from numpy.typing import NDArray

from src.utils import BoundingBox, SuperVisionPlot, BoxPlotType, KeyPointBox, KeyPointPlotType


class ActionDetector:
    def __init__(self, cfg):
        self.model = YOLO(cfg['weight'])
        self.labels = {i: n for i, n in enumerate(self.model.names)}

    def predict(self, inputs: NDArray, verbose=False) -> List[BoundingBox]:
        outputs = self.model.predict(inputs, verbose=verbose)
        confs = outputs[0].boxes.conf.cpu().detach().numpy().tolist()
        boxes = outputs[0].boxes.xyxy.cpu().detach().numpy().tolist()
        classes = outputs[0].boxes.cls.cpu().detach().numpy().astype(int).tolist()

        results = []
        for box, conf, cl in zip(boxes, confs, classes):
            name = self.labels[cl]
            b = BoundingBox(box, name=name, conf=float(conf), label=cl)
            results.append(b)
        return results

    def batch_predict(self, inputs: List[NDArray], verbose=False) -> List[List[BoundingBox]]:
        outputs = self.model.predict(inputs, verbose=verbose)

        results = []
        for output in outputs:
            confs = output.boxes.conf.cpu().detach().numpy().tolist()
            boxes = output.boxes.xyxy.cpu().detach().numpy().tolist()
            classes = output.boxes.cls.cpu().detach().numpy().astype(int).tolist()

            temp = []
            for box, conf, cl in zip(boxes, confs, classes):
                name = self.labels[cl]
                b = BoundingBox(box, name=name, conf=float(conf), label=cl)
                temp.append(b)
            results.append(temp)
        return results

    @staticmethod
    def keep(bboxes: List[BoundingBox], label: int) -> List[BoundingBox]:
        return [bbox for bbox in bboxes if bbox.label == label]

    @staticmethod
    def filter(bboxes, label: int):
        return [bbox for bbox in bboxes if bbox.label != label]

    @staticmethod
    def draw(input_frame: NDArray, items: List[BoundingBox | KeyPointBox]):
        if not len(items):
            return input_frame

        if isinstance(items[0], BoundingBox):
            input_frame = SuperVisionPlot.bbox_plot(input_frame, items, plot_type=BoxPlotType.Corner)
        else:
            input_frame = SuperVisionPlot.keypoint_plot(input_frame, items, plot_type=KeyPointPlotType.Vertex)

        return input_frame


# if __name__ == '__main__':
#     import os
#     import cv2
#     from tqdm import tqdm
#     from pathlib import Path
#     video = '/home/masoud/Desktop/projects/volleyball_analytics/data/raw/videos/test/videos/11_short.mp4'
#     output = '/home/masoud/Desktop/projects/volleyball_analytics/runs/inference/det'
#     os.makedirs(output, exist_ok=True)
#     cfg = {
#         'weight': '/home/masoud/Desktop/projects/volleyball_analytics/weights/vb_actions_6_class/model1/weights/best.pt'
#     }
#
#     action_detector = ActionDetector(cfg=cfg)
#     cap = cv2.VideoCapture(video)
#     assert cap.isOpened()
#
#     w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_file = Path(output) / (Path(video).stem + '_output.mp4')
#     writer = cv2.VideoWriter(output_file.as_posix(), fourcc, fps, (w, h))
#
#     for fno in tqdm(list(range(n_frames))):
#         cap.set(1, fno)
#         status, frame = cap.read()
#         bboxes = action_detector.predict(frame)
#         frame = action_detector.draw(frame, bboxes)
#         writer.write(frame)
#
#     cap.release()
#     writer.release()
#     print(f'saved results in {output_file}')
#     cv2.destroyAllWindows()
#

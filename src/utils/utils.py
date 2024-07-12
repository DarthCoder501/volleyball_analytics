import cv2
import numpy as np
from time import time

from tqdm import tqdm
from pathlib import Path
from functools import wraps


def video_write(input: str, output_path: str, yolo_model, config):
    action_detector = yolo_model(cfg=config)
    cap = cv2.VideoCapture(input)
    assert cap.isOpened()

    w, h, fps, _, n_frames = [int(cap.get(i)) for i in range(3, 8)]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = Path(output_path) / (Path(input).stem + '_output.mp4')
    writer = cv2.VideoWriter(output_file.as_posix(), fourcc, fps, (w, h))

    for fno in tqdm(list(range(n_frames))):
        cap.set(1, fno)
        status, frame = cap.read()
        bboxes = action_detector.predict(frame)
        frame = action_detector.draw(frame, bboxes)
        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'saved results in {output_file}')


def timeit(f):
    """A wrapper around function f that measures the execution time."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te - ts: .2f} sec')
        return result

    return wrap


def state_summarize(states):
    curr = states[0]
    temp = [curr]
    for item in states:
        if item == curr:
            continue
        else:
            curr = item
            temp.append(curr)
    return temp


# class ColorPalette:
#     white = (255, 255, 255)
#     black = (0, 0, 0)
#     purple = (148, 0, 211)
#     magenta = (255, 0, 255)
#     blue = (0, 0, 255)
#     green = (0, 255, 0)
#     yellow = (255, 215, 0)
#     orange = (255, 140, 0)
#     brown = (205, 133, 63)
#     pink = (240, 128, 128)
#     red = (255, 0, 0)
#     aqua = (0, 255, 255)
#     grey = (128, 128, 128)
#
#     bgr_purple = (211, 0, 148)
#     bgr_blue = (255, 0, 0)
#     bgr_red = (0, 0, 255)
#     bgr_orange = (0, 140, 255)
#     bgr_yellow = (0, 215, 255)
#     bgr_pink = (128, 128, 240)
#     bgr_brown = (63, 133, 205)
#     bgr_aqua = (255, 255, 0)


class CourtCoordinates:
    def __init__(self, points: dict):
        self.court = np.array(points['court'], dtype=np.int32)
        self.front = np.array(points['attack'], dtype=np.int32)
        self.center_line = np.array(points['center'], dtype=np.int32)
        self.net = np.array(points['net'], dtype=np.int32)

    def is_inside_main_zone(self, point: tuple):
        result = cv2.pointPolygonTest(self.court, point, False)
        return result > 0

    def is_inside_front_zone(self, point: tuple):
        result = cv2.pointPolygonTest(self.front, point, False)
        return result > 0


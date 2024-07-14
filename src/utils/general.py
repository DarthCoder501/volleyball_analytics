import cv2
import numpy as np
from time import time

from supervision.draw.color import ROBOFLOW_COLOR_PALETTE
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


class CourtCoordinates:
    def __init__(self, points: dict):
        self.court = np.array(points['court_segmentation'], dtype=np.int32)
        self.front = np.array(points['attack'], dtype=np.int32)
        self.center_line = np.array(points['center'], dtype=np.int32)
        self.net = np.array(points['net'], dtype=np.int32)

    def is_inside_main_zone(self, point: tuple):
        result = cv2.pointPolygonTest(self.court, point, False)
        return result > 0

    def is_inside_front_zone(self, point: tuple):
        result = cv2.pointPolygonTest(self.front, point, False)
        return result > 0


def get_color(name):
    colormap = ROBOFLOW_COLOR_PALETTE
    match name:
        case 'ball':
            color = colormap[0]
        case 'person':
            color = colormap[1]
        case 'spike':
            color = colormap[2]
        case 'block':
            color = colormap[3]
        case 'set':
            color = colormap[4]
        case 'receive':
            color = colormap[5]
        case 'serve':
            color = colormap[6]
        case 'court':
            color = colormap[7]
        case 'team_up':
            color = colormap[8]
        case 'team_down':
            color = colormap[9]
        case _:
            print(f"item name not included here: {name}")
            color = colormap[12]

    return color


def get_class_id(name):
    match name:
        case 'ball':
            index = 0
        case 'person':
            index = 1
        case 'spike':
            index = 2
        case 'block':
            index = 3
        case 'set':
            index = 4
        case 'receive':
            index = 5
        case 'serve':
            index = 6
        case 'court':
            index = 7
        case 'team_up':
            index = 8
        case 'team_down':
            index = 9
        case _:
            print(f"item name not included here: {name}")
            index = 12

    return index

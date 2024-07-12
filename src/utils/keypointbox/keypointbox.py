import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils import BoundingBox, ColorPalette, sv_plot, PlotType


class KeyPointBox:
    def __init__(self, keypoints: NDArray, conf: float = 0, name: str = None):
        """Single players keypoints from single yolo. A frame will have a list of these objects.
        Args:
            keypoints:
            name:
        """
        self.name = name
        self.keypoints = keypoints.astype(int)
        self.conf = conf
        self.colors = np.random.randint(low=0, high=255, size=(7, 3))

        self.pose_pairs = {
            'green': [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
            'orange': [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],
            'purple': [(5, 11), (6, 12), (11, 12)],
            'blue': [(11, 13), (12, 14), (13, 15), (14, 16)]
        }
        self.box = self.get_bbox()

    def get_bbox(self):
        """Generates the BoundingBox for keypoints."""
        height_margin = 10
        width_margin = 10
        xs = self.keypoints[:, 0]
        xs = xs[xs != 0]
        ys = self.keypoints[:, 1]
        ys = ys[ys != 0]

        if not len(xs) or not len(ys):
            return None

        min_x = np.min(xs) - width_margin if (np.min(xs) - width_margin) > 0 else np.min(xs)
        min_y = np.min(ys) - height_margin if (np.min(xs) - height_margin) > 0 else np.min(xs)
        max_x = np.max(xs) + width_margin
        max_y = np.max(ys) + height_margin

        return BoundingBox([min_x, min_y, max_x, max_y], name=self.name, conf=self.conf)

    @property
    def nose(self):
        return tuple(self.keypoints[0])

    @property
    def left_eye(self):
        return tuple(self.keypoints[1])

    @property
    def right_eye(self):
        return tuple(self.keypoints[2])

    @property
    def left_ear(self):
        return tuple(self.keypoints[3])

    @property
    def right_ear(self):
        return tuple(self.keypoints[4])

    @property
    def left_shoulder(self):
        return tuple(self.keypoints[5])

    @property
    def right_shoulder(self):
        return tuple(self.keypoints[6])

    @property
    def left_elbow(self):
        return tuple(self.keypoints[7])

    @property
    def right_elbow(self):
        return tuple(self.keypoints[8])

    @property
    def left_wrist(self):
        return tuple(self.keypoints[9])

    @property
    def right_wrist(self):
        return tuple(self.keypoints[10])

    @property
    def left_hip(self):
        return tuple(self.keypoints[11])

    @property
    def right_hip(self):
        return tuple(self.keypoints[12])

    @property
    def left_knee(self):
        return tuple(self.keypoints[13])

    @property
    def right_knee(self):
        return tuple(self.keypoints[14])

    @property
    def left_ankle(self):
        return tuple(self.keypoints[15])

    @property
    def right_ankle(self):
        return tuple(self.keypoints[16])

    @property
    def center(self):
        return self.get_bbox().center

    @staticmethod
    def is_kp_detected(kp):
        """
        In yolo-v8 when the kp is not detected, it returns 0, 0 for x, y ...
        Args:
            kp:

        Returns:

        """
        return kp[0] != 0 and kp[1] != 0

    @property
    def is_facing_to_camera(self):
        lw = self.left_wrist
        rw = self.right_wrist

        le = self.left_elbow
        re = self.right_elbow

        ls = self.left_shoulder
        rs = self.right_shoulder

        la = self.left_ankle
        ra = self.right_ankle

        lk = self.left_knee
        rk = self.right_knee

        if self.is_kp_detected(lw) and self.is_kp_detected(rw):
            return lw[0] > rw[0]
        elif self.is_kp_detected(le) and self.is_kp_detected(re):
            return le[0] > re[0]
        elif self.is_kp_detected(lk) and self.is_kp_detected(rk):
            return lk[0] > rk[0]
        elif self.is_kp_detected(ls) and self.is_kp_detected(rs):
            return ls[0] > rs[0]
        elif self.is_kp_detected(la) and self.is_kp_detected(ra):
            return la[0] > ra[0]

    def draw_lines(self, img: NDArray) -> NDArray:
        for color, pairs in self.pose_pairs.items():
            for pair in pairs:
                pt1 = self.keypoints[pair[0]]
                pt2 = self.keypoints[pair[1]]
                if self.is_kp_detected(pt1) and self.is_kp_detected(pt2):
                    match color:
                        case 'green':
                            cv2.line(img, pt1, pt2, ColorPalette.green, 2)
                        case 'orange':
                            cv2.line(img, pt1, pt2, ColorPalette.orange, 2)
                        case 'purple':
                            cv2.line(img, pt1, pt2, ColorPalette.purple, 2)
                        case 'blue':
                            cv2.line(img, pt1, pt2, ColorPalette.blue, 2)
        return img

    def draw_ellipse(self, img: NDArray, color: tuple = ColorPalette.blue) -> NDArray:
        return sv_plot(image=img, bboxes=[self.box], color=color, plot_type=PlotType.Ellipse)

    def draw_marker(self, img: np.ndarray, color: tuple = ColorPalette.green) -> np.ndarray:
        return sv_plot(image=img, bboxes=[self.box], color=color, plot_type=PlotType.Triangle)

    def json(self):
        # TODO: Needs integration with self.is_kp_detected...
        js = {
            'nose': self.nose,
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'left_ear': self.left_ear,
            'right_ear': self.right_ear,
            'left_shoulder': self.left_shoulder,
            'right_shoulder': self.right_shoulder,
            'left_elbow': self.left_elbow,
            'right_elbow': self.right_elbow,
            'left_wrist': self.left_wrist,
            'right_wrist': self.right_wrist,
            'left_hip': self.left_hip,
            'right_hip': self.right_hip,
            'left_knee': self.left_knee,
            'right_knee': self.right_knee,
            'left_ankle': self.left_ankle,
            'right_ankle': self.right_ankle
        }
        return js

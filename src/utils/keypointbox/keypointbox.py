import numpy as np
from numpy.typing import NDArray

from src.utils import BoundingBox


class KeyPointBox:
    __slots__ = ['name', "label", "np_key_points", "key_points", "conf", "box"]

    def __init__(self, key_points: NDArray, conf: float = 0, name: str = None, label: int = None):
        """Single player_detection keypoints from single yolo. A frame will have a list of these objects.
        Args:
            key_points:
            name:
        """
        self.name = name
        self.label = label
        self.np_key_points = key_points
        self.key_points = key_points.astype(int).tolist()
        self.conf = conf
        self.box = self.get_bbox()

    def get_kps(self) -> NDArray:
        return self.np_key_points

    def get_bbox(self) -> BoundingBox | None:
        """Generates the BoundingBox for keypoints."""
        height_margin = 10
        width_margin = 10
        xs = self.key_points[:, 0]
        xs = xs[xs != 0]
        ys = self.key_points[:, 1]
        ys = ys[ys != 0]

        if not len(xs) or not len(ys):
            return None

        min_x = np.min(xs) - width_margin if (np.min(xs) - width_margin) > 0 else np.min(xs)
        min_y = np.min(ys) - height_margin if (np.min(xs) - height_margin) > 0 else np.min(xs)
        max_x = np.max(xs) + width_margin
        max_y = np.max(ys) + height_margin

        return BoundingBox([min_x, min_y, max_x, max_y], name=self.name, conf=self.conf)

    @property
    def nose(self) -> tuple:
        return tuple(self.key_points[0])

    @property
    def left_eye(self) -> tuple:
        return tuple(self.key_points[1])

    @property
    def right_eye(self) -> tuple:
        return tuple(self.key_points[2])

    @property
    def left_ear(self) -> tuple:
        return tuple(self.key_points[3])

    @property
    def right_ear(self) -> tuple:
        return tuple(self.key_points[4])

    @property
    def left_shoulder(self) -> tuple:
        return tuple(self.key_points[5])

    @property
    def right_shoulder(self) -> tuple:
        return tuple(self.key_points[6])

    @property
    def left_elbow(self) -> tuple:
        return tuple(self.key_points[7])

    @property
    def right_elbow(self) -> tuple:
        return tuple(self.key_points[8])

    @property
    def left_wrist(self) -> tuple:
        return tuple(self.key_points[9])

    @property
    def right_wrist(self) -> tuple:
        return tuple(self.key_points[10])

    @property
    def left_hip(self) -> tuple:
        return tuple(self.key_points[11])

    @property
    def right_hip(self) -> tuple:
        return tuple(self.key_points[12])

    @property
    def left_knee(self) -> tuple:
        return tuple(self.key_points[13])

    @property
    def right_knee(self) -> tuple:
        return tuple(self.key_points[14])

    @property
    def left_ankle(self) -> tuple:
        return tuple(self.key_points[15])

    @property
    def right_ankle(self) -> tuple:
        return tuple(self.key_points[16])

    @property
    def center(self) -> tuple:
        return self.get_bbox().center

    @staticmethod
    def is_kp_detected(kp) -> bool:
        """
        In yolo-v8 when the kp is not detected, it returns 0, 0 for x, y ...
        Args:
            kp:

        Returns:

        """
        return kp[0] != 0 and kp[1] != 0

    @property
    def is_facing_to_camera(self) -> bool:
        """
        Whether a person (based on pose keypoints) is facing the camera or backward to camera.
        It is used to find out which players belong to which teams in volleyball from behind camera.

        Returns:

        """
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

    def json(self) -> dict:
        js = {
            'nose': self.nose if self.is_kp_detected(self.nose) else None,
            'left_eye': self.left_eye if self.is_kp_detected(self.left_eye) else None,
            'right_eye': self.right_eye if self.is_kp_detected(self.right_eye) else None,
            'left_ear': self.left_ear if self.is_kp_detected(self.left_ear) else None,
            'right_ear': self.right_ear if self.is_kp_detected(self.right_ear) else None,
            'left_shoulder': self.left_shoulder if self.is_kp_detected(self.left_shoulder) else None,
            'right_shoulder': self.right_shoulder if self.is_kp_detected(self.right_shoulder) else None,
            'left_elbow': self.left_elbow if self.is_kp_detected(self.left_elbow) else None,
            'right_elbow': self.right_elbow if self.is_kp_detected(self.right_elbow) else None,
            'left_wrist': self.left_wrist if self.is_kp_detected(self.left_wrist) else None,
            'right_wrist': self.right_wrist if self.is_kp_detected(self.right_wrist) else None,
            'left_hip': self.left_hip if self.is_kp_detected(self.left_hip) else None,
            'right_hip': self.right_hip if self.is_kp_detected(self.right_hip) else None,
            'left_knee': self.left_knee if self.is_kp_detected(self.left_knee) else None,
            'right_knee': self.right_knee if self.is_kp_detected(self.right_knee) else None,
            'left_ankle': self.left_ankle if self.is_kp_detected(self.left_ankle) else None,
            'right_ankle': self.right_ankle if self.is_kp_detected(self.right_ankle) else None,
            'team': 'up' if self.is_facing_to_camera else "down"
        }
        return js

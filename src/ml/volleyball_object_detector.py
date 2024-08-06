from typing import List, Tuple

import numpy as np
# import yaml
import json
from numpy.typing import NDArray
# from yaml.loader import SafeLoader

from src.utils import BoundingBox, KeyPointBox, KeyPointPlotType, BoxPlotType, SuperVisionPlot
from src.ml.ball_detection import BallSegmentor
from src.ml.action_detection import ActionDetector
from src.ml.player_detection import PlayerSegmentor, PlayerDetector, PoseEstimator


class VolleyBallObjectDetector:
    def __init__(self, config: dict, video_name: str = None, use_player_detection=True):
        self.config = config
        court_dict = None
        # TODO: make it work even if there is no court_segmentation json for the specific match....
        if video_name is not None:
            try:
                court_dict = json.load(open(self.config['court_json']))[video_name]
            except KeyError:
                court_dict = None
        if use_player_detection:
            self.player_detector = PlayerDetector(self.config['yolo']['player_detection'], court_dict=court_dict)
        else:
            self.player_detector = PlayerSegmentor(self.config['yolo']['player_segmentation'], court_dict=court_dict)
        self.action_detector = ActionDetector(self.config['yolo']['action_detection'])
        self.ball_detector = BallSegmentor(self.config['yolo']['ball_segmentation'])
        self.pose_estimator = PoseEstimator(self.config['yolo']['pose_estimation'])

    # TODO: FIXME: Make code adaptable to batch processing ...
    def detect_balls(self, inputs: NDArray | List[NDArray]):
        if isinstance(inputs, np.ndarray):
            return self.ball_detector.predict(inputs=inputs)
        return self.ball_detector.batch_predict(inputs=inputs)

    def detect_actions(self, inputs: NDArray | List[NDArray]):
        if isinstance(inputs, np.ndarray):
            return self.action_detector.predict(inputs=inputs)
        return self.action_detector.batch_predict(inputs=inputs)

    def detect_keypoints(self, inputs: NDArray | List[NDArray]):
        if isinstance(inputs, np.ndarray):
            return self.pose_estimator.predict(inputs=inputs)
        return self.pose_estimator.batch_predict(inputs=inputs)

    def segment_players(self, inputs: NDArray | List[NDArray]):
        return self.player_detector.predict(input_frame=inputs)

    @staticmethod
    def keep(bboxes: List[BoundingBox], to_keep: List[str] | Tuple[str] | str) -> List[BoundingBox]:
        if isinstance(to_keep, list) or isinstance(to_keep, tuple):
            objects = [bbox for bbox in bboxes if bbox.name in to_keep]
        elif isinstance(to_keep, str):
            objects = [bbox for bbox in bboxes if bbox.name == to_keep]
        else:
            raise ValueError('the to_keep parameter type is wrong')

        objects.sort(key=lambda x: x.conf)
        return objects

    @staticmethod
    def filter(bboxes: List[BoundingBox], to_filter: List[str] | Tuple[str] | str):
        if isinstance(to_filter, list) or isinstance(to_filter, tuple):
            objects = [bbox for bbox in bboxes if bbox.name not in to_filter]
        elif isinstance(to_filter, str):
            objects = [bbox for bbox in bboxes if bbox.name != to_filter]
        else:
            raise ValueError('the to_filter parameter type is wrong')
        objects.sort(key=lambda x: x.name)
        return objects

    def draw_bboxes(self, image, bboxes: List[BoundingBox | KeyPointBox], bb_plot_type: int = BoxPlotType.Corner,
                    kp_plot_type: int = KeyPointPlotType.Ellipse) -> NDArray:
        image = SuperVisionPlot.bbox_plot(image, bboxes=[bb for bb in bboxes if isinstance(bb, BoundingBox)],
                                          plot_type=bb_plot_type)
        image = SuperVisionPlot.keypoint_plot(image, key_points=[kp for kp in bboxes if isinstance(kp, KeyPointBox)],
                                              plot_type=kp_plot_type)

        image = self.action_detector.draw(input_frame=image, items=bboxes)
        return image


# if __name__ == '__main__':
#     config_file = '/home/masoud/Desktop/projects/volleyball_analytics/conf/ml_models.yaml'
#     setup = '/home/masoud/Desktop/projects/volleyball_analytics/conf/setup.yaml'
#     court_json = '/home/masoud/Desktop/projects/volleyball_analytics/conf/reference_pts.json'
#     video_name = "22.mp4"
#
#     cfg: dict = yaml.load(open(config_file), Loader=SafeLoader)
#     cfg2: dict = yaml.load(open(setup), Loader=SafeLoader)
#     cfg.update(cfg2)
#
#     vb_detector = VolleyBallObjectDetector(cfg, video_name)

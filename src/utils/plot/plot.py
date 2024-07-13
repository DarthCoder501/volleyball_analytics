from typing import List

import numpy as np
import supervision as sv
from numpy.typing import NDArray
from supervision import Color

from src.utils import BoundingBox, KeyPointBox, BoxPlotType, KeyPointPlotType


class SuperVisionPlot:
    @staticmethod
    def bbox_plot(image: NDArray, bboxes: List[BoundingBox], color: tuple = None,
                  plot_type: int = BoxPlotType.Color, draw_label: bool = False,
                  draw_name: bool = False, draw_percentage: bool = False, ) -> NDArray:
        if not len(bboxes):
            return image

        if color is None:
            c = sv.ColorPalette.DEFAULT
        else:
            c = sv.Color(r=color[0], g=color[1], b=color[2])
        match plot_type:
            case BoxPlotType.Triangle:
                annotator = sv.TriangleAnnotator(color=c)
            case BoxPlotType.Corner:
                annotator = sv.BoxCornerAnnotator(color=c, thickness=2)
            case BoxPlotType.Dot:
                annotator = sv.DotAnnotator(color=c, radius=10)
            case BoxPlotType.Circle:
                annotator = sv.CircleAnnotator(color=c, thickness=2)
            case BoxPlotType.Color:
                annotator = sv.ColorAnnotator(color=c, opacity=0.4)
            case BoxPlotType.Round:
                annotator = sv.RoundBoxAnnotator(color=c, thickness=2)
            case BoxPlotType.Ellipse:
                annotator = sv.EllipseAnnotator(color=c, thickness=2)
            case BoxPlotType.Bar:
                annotator = sv.PercentageBarAnnotator(color=c)
            case None | _:
                annotator = sv.BoundingBoxAnnotator(color=c, thickness=3)

        detections = sv.Detections(
            xyxy=np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes]),
            class_id=np.array([bbox.label for bbox in bboxes]),
            confidence=np.array([bbox.conf for bbox in bboxes])
        )
        image = annotator.annotate(scene=image, detections=detections)
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_color=sv.Color(r=4, g=4, b=4),
            text_thickness=1,
            color=c
        )

        if draw_label:
            if draw_name:
                labels = [f"{bbox.label}-{bbox.name}-{bbox.conf: 0.3f}" for bbox in bboxes]
            else:
                labels = [f"{bbox.label}-{bbox.conf: 0.3f}" for bbox in bboxes]
            image = label_annotator.annotate(
                scene=image,
                detections=detections,
                labels=labels
            )

        if draw_percentage:
            percentage_annotator = sv.PercentageBarAnnotator(
                color=c,
                position=sv.Position.BOTTOM_CENTER,
                border_color=Color.RED
            )
            image = percentage_annotator.annotate(scene=image, detections=detections)
        return image

    @staticmethod
    def keypoint_plot(image: NDArray, key_points: List[KeyPointBox], plot_type: int = KeyPointPlotType.VertexLabel):
        if not len(key_points):
            return image
        # TODO: Differentiate between bbox plots and keypoints plot
        # TODO: Add bbox triangle and ellipse plots....
        c = sv.ColorPalette.DEFAULT
        match plot_type:
            case KeyPointPlotType.Vertex:
                annotator = sv.VertexAnnotator(color=c)
            case KeyPointPlotType.VertexLabel:
                annotator = sv.VertexLabelAnnotator(color=c, text_thickness=2)
            case None | _:
                annotator = sv.EdgeAnnotator(color=c, thickness=2)

        sv_keypoints = sv.KeyPoints(
            xy=np.array([kp.get_kps() for kp in key_points]),
            class_id=np.array([bbox.label for bbox in key_points]),
            confidence=np.array([bbox.conf for bbox in key_points])
        )
        image = annotator.annotate(scene=image, key_points=sv_keypoints)
        return image

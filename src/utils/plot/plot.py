from typing import List

import numpy as np
import supervision as sv
from numpy.typing import NDArray
from supervision import Color

from src.utils import BoundingBox, PlotType


def sv_plot(
        image: NDArray,
        bboxes: List[BoundingBox],
        color: tuple = None,
        plot_type: int = PlotType.Color,
        draw_label: bool = False,
        draw_name: bool = False,
        draw_percentage: bool = False,
        is_text: bool = False
) -> NDArray:
    if not len(bboxes):
        return image

    if color is None:
        c = sv.ColorPalette.DEFAULT
    else:
        c = sv.Color(r=color[0], g=color[1], b=color[2])
    match plot_type:
        case PlotType.Triangle:
            annotator = sv.TriangleAnnotator(color=c)
        case PlotType.Corner:
            annotator = sv.BoxCornerAnnotator(color=c, thickness=2)
        case PlotType.Dot:
            annotator = sv.DotAnnotator(color=c, radius=10)
        case PlotType.Circle:
            annotator = sv.CircleAnnotator(color=c, thickness=2)
        case PlotType.Color:
            annotator = sv.ColorAnnotator(color=c, opacity=0.4)
        case PlotType.Round:
            annotator = sv.RoundBoxAnnotator(color=c, thickness=2)
        case PlotType.Ellipse:
            annotator = sv.EllipseAnnotator(color=c, thickness=2)
        case PlotType.Bar:
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
    if is_text:
        titles = []
        for bbox in bboxes:
            att = bbox.attributes.get("text", None)
            conf = f"{bbox.conf: 0.4f}"
            title = f"text: {att if att is not None else 'not added'} | {conf}"
            titles.append(title)
        image = label_annotator.annotate(
            scene=image,
            detections=detections,
            labels=titles
        )
        return image

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

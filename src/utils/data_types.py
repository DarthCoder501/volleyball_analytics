from enum import IntEnum


class DatasetType(IntEnum):
    NoValue = 0
    YoloDatasetType = 1
    COCODatasetType = 2


class BoxPlotType(IntEnum):
    Color = 1
    Triangle = 2
    Corner = 3
    Dot = 4
    Circle = 5
    Ellipse = 6
    Bar = 7
    Round = 8


class KeyPointPlotType(IntEnum):
    Vertex = 0
    Edge = 1
    VertexLabel = 2
    Ellipse = 3
    Triangle = 4

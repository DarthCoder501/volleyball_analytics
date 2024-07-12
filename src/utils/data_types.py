from enum import IntEnum


class DatasetType(IntEnum):
    NoValue = 0
    YoloDatasetType = 1
    COCODatasetType = 2


class PlotType(IntEnum):
    Color = 1
    Triangle = 2
    Corner = 3
    Dot = 4
    Circle = 5
    Ellipse = 6
    Bar = 7
    Round = 8

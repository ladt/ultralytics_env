import importlib
_real = importlib.import_module("ultralytics")

from ultralytics_extension.models.yolo import YOLO as _YOLO

def __getattr__(name):
    if name == "YOLO":
        return _YOLO
    return getattr(_real, name)

def __dir__():
    return list(_real.__all__) + ["YOLO"]

__all__ = list(_real.__all__) + ["YOLO"]

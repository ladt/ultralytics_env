from ultralytics.models.yolo.model import YOLO as _YOLO

from ultralytics_extension.models.yolo.classify import (
    ClassificationValidator, ClassificationTrainer, ClassificationPredictor
)

class YOLO(_YOLO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def task_map(self):
        parent_map: dict = super().task_map
        tm = parent_map.copy()
        classify_cfg = tm["classify"].copy()
        classify_cfg["trainer"] = ClassificationTrainer
        classify_cfg["validator"] = ClassificationValidator
        classify_cfg["predictor"] = ClassificationPredictor
        tm["classify"] = classify_cfg
        return tm

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def val(self, *args, **kwargs):
        return super().val(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)
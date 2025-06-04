from ultralytics.models.yolo.classify.val import ClassificationValidator as _ClassificationValidator

from ultralytics_extension.data import ClassificationDataset
from ultralytics_extension.utils.metrics import ClassifyMetrics

class ClassificationValidator(_ClassificationValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = ClassifyMetrics()

    def init_metrics(self, model):
        super().init_metrics(model)

    def build_dataset(self, img_path):
        return ClassificationDataset(
            root=img_path,
            args=self.args,
            augment=False,
            prefix=self.args.split,
        )

from ultralytics.models.yolo.classify.predict import ClassificationPredictor as ClassificationPredictor_

from ultralytics_extension.data.augment import classify_transforms

class ClassificationPredictor(ClassificationPredictor_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_source(self, source):
        super().setup_source(source)
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0]),
            )
        )

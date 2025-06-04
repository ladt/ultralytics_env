from copy import copy


from ultralytics.models.yolo.classify.train import ClassificationTrainer as _ClassificationTrainer

from ultralytics_extension.data import ClassificationDataset
from .val import ClassificationValidator

class ClassificationTrainer(_ClassificationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_model_attributes(self):
        super().set_model_attributes()


    def get_validator(self):
        self.loss_names = ["loss"]
        return ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

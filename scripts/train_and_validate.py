from pathlib import Path
import platform

import cv2

from ultralytics_extension import YOLO

if platform.system() == "Windows":
    PLATFORM_PREFIX = "//devbitshares.devbit.io/aerospace_cvai"
elif platform.system() == "Linux":
    PLATFORM_PREFIX = "/media/aerospace_cvai"

PROJECT_ROOT = Path(__file__).parents[1].resolve()

model_scale = "n"

model_yaml_path = PROJECT_ROOT / "configs" / "model" / f"yolo11{model_scale}-cls_modernization.yaml"
model_pt_path = Path(PLATFORM_PREFIX) / "shared_weights" / "Ultralytics" / "pretrained_classification_models" / f"yolo11{model_scale}-cls.pt"
train_yaml_path = PROJECT_ROOT / "configs" / "train" / f"classify_modernization.yaml"

project_name = Path(PLATFORM_PREFIX) / "shared_outputs" / "classifier_training"

dataset_name = '8_16'  # '16-20' # '8-12'
exp_name = f'vehicle_classification_{dataset_name}_pad_random'

data_dir = Path(PLATFORM_PREFIX) / "datasets" / "ultralytics" / "skeye_classifier_flat_datasets" / dataset_name
# data_dir = Path(PLATFORM_PREFIX) / "datasets" / "ultralytics" / "demo_dataset"

train = True
validate = False
predict = False
imgsz = 64
# name_order = ["motorcycle", "private", "pick-up", "van", "bus", "truck", "tractor"]

def main():
    # Training and validation logic
    if train:
        model = YOLO(model_yaml_path, task='classify')  # build from YAML
        model.load(model_pt_path)  # transfer weights
        # Train the classifier
        model.train(
            data=data_dir,
            cfg=train_yaml_path,
            epochs=1000,
            project=project_name,
            name=exp_name,
            imgsz=imgsz,
            batch=8000,
        )

    if validate or predict:
        model = YOLO(model_yaml_path, task='classify')  # build from YAML
        # Validate on the same dataset
        best_weights_path = project_name / f"{exp_name}" / "weights" / "best.pt"
        model.load(best_weights_path).to('cuda')  # transfer weights

        if validate:
            metrics = model.val(
                data=data_dir,
                imgsz=imgsz,
                batch=6000,
                project=project_name,
                name=f'{exp_name}_validation',
            )
        
        if predict:
            img_path = Path(PLATFORM_PREFIX) / "datasets/ultralytics/skeye_classifier_flat_datasets/8_16/val/pick-up/SkeyeNetanyaKashiotSingleFrames_4_39165616_0001_idx_3737.jpg"
            assert img_path.exists()
            img = cv2.imread(str(img_path))
            results = model.predict(img ,imgsz=imgsz)


if __name__ == "__main__":
    main()

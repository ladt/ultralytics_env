from ultralytics_extension.data.augment import classify_transforms, classify_augmentations
from ultralytics.data.dataset import ClassificationDataset as _ClassificationDataset

from .augment import classify_augmentations

class ClassificationDataset(_ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment=augment, prefix=prefix)

        if augment:
            pad_value = 0
            pad_random = True
            disable_color_jitter = True

            scale = (1.0 - args.scale, 1.0)
            self.torch_transforms = classify_augmentations(
                    size=args.imgsz,
                    scale=scale,
                    hflip=args.fliplr,
                    vflip=args.flipud,
                    erasing=args.erasing,
                    auto_augment=args.auto_augment,
                    hsv_h=args.hsv_h,
                    hsv_s=args.hsv_s,
                    hsv_v=args.hsv_v,
                    pad_value=pad_value,
                    pad_random=pad_random,
                    disable_color_jitter=disable_color_jitter
                )
        else:
            self.torch_transforms = classify_transforms(size=args.imgsz)

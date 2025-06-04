import functools

import cv2
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torchvision.transforms import Compose as TCompose

from ultralytics.data.augment import classify_augmentations as _classify_augmentations, classify_transforms as _classify_transforms

class AlbToPil:
    def __init__(self, pipeline: A.Compose):
        self.pipeline = pipeline

    def __call__(self, img):
        # PIL RGB -> numpy
        arr = np.array(img)
        # geometric augmentations (still RGB)
        img = self.pipeline(image=arr)["image"]
        # back to PIL RGB
        return Image.fromarray(img)

def _make_geom_pipeline(size, pad_value: int = 0, pad_random: bool = False, interpolation: str ="BILINEAR"):
    """
    create an Albumentations pipeline:
        - if square: LongestMaxSize + PadIfNeeded to (size, size)
        - if rectangle: Resize to (h, w) exactly
    """
    if isinstance(size, (list, tuple)) and len(size) == 2:
        h, w = size
    else:
        h = w = size
    cv2_interp = {"NEAREST": 0, "BILINEAR": 1, "BICUBIC": 2}[interpolation]
    pos = "random" if pad_random else "center"
    if h == w:
        return A.Compose([
            A.LongestMaxSize(max_size=h, interpolation=cv2_interp),
            A.PadIfNeeded(min_height=h, min_width=h, position=pos, border_mode=cv2.BORDER_CONSTANT, fill=pad_value)
        ])
    else:
        return A.Compose([
            A.Resize(height=h, width=w, interpolation=cv2_interp),
        ])

'''
 - Predictor (preprocess function ultralytics/models/yolo/classify/predict.py -> process function):
    * loads BGR -> converts to RGB -> converts to PIL image -> applies Callable transform
- other call to classify_augmentations/classifiy_transforms: ultralytics/data/dataset.py -> ClassificationDataset
'''

@functools.wraps(_classify_augmentations)
def classify_augmentations(*args, ratio=(1.0, 1.0), pad_value: int = 0, pad_random: bool = False, disable_color_jitter: bool = False, **kwargs):
    kwargs.setdefault('ratio', ratio) # enable future use in 'ratio' kwarg
    if len(args) > 0:
        size = args[0]
    else:
        size = kwargs.get("size")
    # map user-provided interpolation name to cv2 flag
    interp = kwargs.get("interpolation", "BILINEAR").upper()

    geometric_ops = _make_geom_pipeline(size, pad_value, pad_random, interpolation=interp)
    to_pil = AlbToPil(geometric_ops)
    orig_transforms: TCompose = _classify_augmentations(*args, **kwargs)
    kept: list = [t for t in orig_transforms.transforms if not isinstance(t, T.RandomResizedCrop)]
    if disable_color_jitter:
        kept: list = [t for t in kept if not isinstance(t, T.ColorJitter)]
    return TCompose([to_pil, *kept])

@functools.wraps(_classify_transforms)
def classify_transforms(*args, pad_value: int = 0, pad_random: bool = False, **kwargs):
    if len(args) > 0:
        size = args[0]
    else:
        size = kwargs.get("size")
    # map user-provided interpolation name to cv2 flag
    interp = kwargs.get("interpolation", "BILINEAR").upper()

    geometric_ops = _make_geom_pipeline(size, pad_value, pad_random, interpolation=interp)
    to_pil = AlbToPil(geometric_ops)
    orig_transforms: TCompose = _classify_transforms(*args, **kwargs)
    kept: list = [t for t in orig_transforms.transforms if not isinstance(t, (T.Resize, T.CenterCrop))]
    return TCompose([to_pil, *kept])

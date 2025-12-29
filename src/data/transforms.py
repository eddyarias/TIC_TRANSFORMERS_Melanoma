from typing import Tuple
import albumentations as A


def get_train_transforms(image_size: int,
                         mean: Tuple[float, float, float],
                         std: Tuple[float, float, float],
                         aug_cfg: dict):
    geo = aug_cfg.get('geometric', {})
    color = aug_cfg.get('color', {})
    reg = aug_cfg.get('regularization', {}).get('coarse_dropout', {})

    transforms = [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=geo.get('hflip_p', 0.5)),
        A.VerticalFlip(p=geo.get('vflip_p', 0.0)),
        # Replace deprecated ShiftScaleRotate with Affine
        A.Affine(translate_percent=(geo.get('shift_limit', 0.1), geo.get('shift_limit', 0.1)),
                 scale=(1.0 - geo.get('scale_limit', 0.1), 1.0 + geo.get('scale_limit', 0.1)),
                 rotate=(-geo.get('rotate_limit', 15), geo.get('rotate_limit', 15)), p=0.7),
        A.ColorJitter(brightness=color.get('brightness_limit', 0.2),
                      contrast=color.get('contrast_limit', 0.2),
                      saturation=color.get('saturation', 0.2),
                      hue=color.get('hue', 0.2), p=0.7),
        # Use GridDropout as robust alternative to CoarseDropout param changes
        A.GridDropout(ratio=0.5, random_offset=True, p=reg.get('p', 0.5)),
        A.Normalize(mean=mean, std=std),
    ]
    return A.Compose(transforms)


def get_val_transforms(image_size: int,
                       mean: Tuple[float, float, float],
                       std: Tuple[float, float, float],
                       aug_cfg: dict):
    center_crop = aug_cfg.get('center_crop', False)
    transforms = []
    if center_crop:
        transforms.append(A.CenterCrop(height=image_size, width=image_size))
    else:
        transforms.append(A.Resize(height=image_size, width=image_size))
    transforms.append(A.Normalize(mean=mean, std=std))
    return A.Compose(transforms)
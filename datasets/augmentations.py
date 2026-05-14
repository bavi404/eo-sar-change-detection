import albumentations as A


def get_train_transforms(crop_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )


def get_eval_transforms(crop_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        ]
    )

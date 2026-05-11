# import random
# import torch
# import torchvision.transforms.functional as F


# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         for t in self.transforms:
#             image, target = t(image, target)
#         return image, target


# class RandomHorizontalFlip:
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             image = F.hflip(image)
#             _, H, W = image.shape
#             # Flip boxes
#             boxes = target["boxes"].clone()
#             boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
#             target["boxes"] = boxes
#             # Flip masks
#             target["masks"] = target["masks"].flip(-1)
#         return image, target


# class RandomVerticalFlip:
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             image = F.vflip(image)
#             _, H, W = image.shape
#             # Flip boxes
#             boxes = target["boxes"].clone()
#             boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
#             target["boxes"] = boxes
#             # Flip masks
#             target["masks"] = target["masks"].flip(-2)
#         return image, target


# class ColorJitter:
#     """Apply random brightness and contrast jitter to image only."""
#     def __init__(self, brightness=0.3, contrast=0.3):
#         self.brightness = brightness
#         self.contrast = contrast

#     def __call__(self, image, target):
#         if random.random() < 0.5:
#             factor = 1.0 + random.uniform(-self.brightness, self.brightness)
#             image = torch.clamp(image * factor, 0.0, 1.0)
#         if random.random() < 0.5:
#             mean = image.mean()
#             factor = 1.0 + random.uniform(-self.contrast, self.contrast)
#             image = torch.clamp((image - mean) * factor + mean, 0.0, 1.0)
#         return image, target


# def get_train_transforms():
#     return Compose([
#         RandomHorizontalFlip(prob=0.5),
#         RandomVerticalFlip(prob=0.5),
#         ColorJitter(brightness=0.3, contrast=0.3),
#     ])


# def get_val_transforms():
#     return None   # no augmentation for val

import random
import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            _, H, W = image.shape
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            target["boxes"] = boxes
            target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            _, H, W = image.shape
            boxes = target["boxes"].clone()
            boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
            target["boxes"] = boxes
            target["masks"] = target["masks"].flip(-2)
        return image, target


class RandomRotation180:
    """Randomly rotate by 0 or 180 degrees.
    180-degree rotation keeps H and W unchanged → no OOM from size change.
    Cells have no preferred orientation so this is a valid augmentation.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        # k=2 → 180° rotation
        image = torch.rot90(image, 2, dims=[1, 2])
        _, H, W = image.shape

        # Rotate masks
        target["masks"] = torch.rot90(target["masks"], 2, dims=[1, 2])

        # Rotate boxes: (x1,y1,x2,y2) → (W-x2, H-y2, W-x1, H-y1)
        boxes = target["boxes"].clone()
        new_boxes = boxes.clone()
        new_boxes[:, 0] = W - boxes[:, 2]
        new_boxes[:, 1] = H - boxes[:, 3]
        new_boxes[:, 2] = W - boxes[:, 0]
        new_boxes[:, 3] = H - boxes[:, 1]
        target["boxes"] = new_boxes

        return image, target


class ColorJitter:
    """Apply random brightness and contrast jitter to image only."""
    def __init__(self, brightness=0.3, contrast=0.3):
        self.brightness = brightness
        self.contrast   = contrast

    def __call__(self, image, target):
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image  = torch.clamp(image * factor, 0.0, 1.0)
        if random.random() < 0.5:
            mean   = image.mean()
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image  = torch.clamp((image - mean) * factor + mean, 0.0, 1.0)
        return image, target


def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        RandomRotation180(prob=0.5),
        ColorJitter(brightness=0.3, contrast=0.3),
    ])


def get_val_transforms():
    return None   # no augmentation for val
# src/dataset.py
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ResNet18 was originally trained on images resized to 224x224,
# with specific normalization values. We match that here so the
# pretrained weights work well with our data.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def get_dataloaders(data_dir="data/raw_sala_sorted", batch_size=32, num_workers=2):
    """
    Loads train and test datasets from data_dir/train and data_dir/test.
    Expects subfolders REAL/ and FAKE/ inside each.
    Returns: train_loader, test_loader, class_names
    """
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_dataset.classes  # e.g. ['FAKE', 'REAL']
    return train_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check: run this file directly to confirm loading works.
    train_loader, test_loader, class_names = get_dataloaders()
    print("Classes found:", class_names)
    print("Number of training images:", len(train_loader.dataset))
    print("Number of test images:", len(test_loader.dataset))

    # Grab one batch and print its shape, just to confirm everything works
    images, labels = next(iter(train_loader))
    print("One batch of images shape:", images.shape)
    print("One batch of labels:", labels[:10])



    '''

    # src/dataset.py

import io
import random

from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ResNet18 was originally trained on images resized to 224x224,
# with specific normalization values. We match that here so the
# pretrained weights work well with our data.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    """
    Re-saves the image as JPEG at a random quality level, in memory.

    Fake-image datasets often carry a consistent compression/generator
    fingerprint. Randomizing JPEG quality during training stops the model
    from keying off "this specific compression signature = fake/real"
    instead of actual visual content.
    """
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownUpResize:
    """
    Randomly downsamples then upsamples the image back to its original size.

    Kills resolution/upsampling-artifact shortcuts (e.g. GAN checkerboard
    patterns) by forcing the model to see the same content at inconsistent
    effective resolutions.
    """
    def __init__(self, scale_range=(0.5, 1.0), p=0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


# Training transform: augmentation designed to destroy low-level
# fingerprints (compression, resolution, generator artifacts) so the
# model is pushed toward learning actual semantic "real vs fake" cues
# instead of shortcutting on dataset-specific quirks.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandomDownUpResize(scale_range=(0.5, 1.0), p=0.3),
    RandomJPEGCompression(quality_range=(30, 95), p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Test/eval transform: NO augmentation. We want test/eval numbers to
# reflect the model's real generalization, not augmented inputs.
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Kept for backwards compatibility with any code importing `transform`
# directly (e.g. predict.py, diagnose_external.py, app.py) - these should
# always use the clean, non-augmented pipeline.
transform = test_transform


def get_dataloaders(data_dir="data/raw_sala_sorted", batch_size=32, num_workers=2):
    """
    Loads train and test datasets from data_dir/train and data_dir/test.
    Expects subfolders REAL/ and FAKE/ inside each.
    Train images get augmentation (see train_transform); test images do not.
    Returns: train_loader, test_loader, class_names
    """
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_dataset.classes  # e.g. ['FAKE', 'REAL']
    return train_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check: run this file directly to confirm loading works.
    train_loader, test_loader, class_names = get_dataloaders()
    print("Classes found:", class_names)
    print("Number of training images:", len(train_loader.dataset))
    print("Number of test images:", len(test_loader.dataset))

    # Grab one batch and print its shape, just to confirm everything works
    images, labels = next(iter(train_loader))
    print("One batch of images shape:", images.shape)
    print("One batch of labels:", labels[:10])
import csv
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

RESIZE_DIM = (600, 600)


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class TomatoesDataset(Dataset):
    """Dataset class used to detect presence of tomatoes on images.
    Each item is a tuple of a image tensor and a class ID (0 = tomato).
    """

    base_transforms = [T.Resize(RESIZE_DIM), T.ToTensor()]

    def __init__(self, imgs_root, annotations_path, label_mapping_path, sampling=None, seed=42):
        # Make sure to get reproducible results
        np.random.seed(seed)

        self.imgs_root = imgs_root

        # Read annotations
        with open(annotations_path) as f:
            self.annotations = json.load(f)

        # Extract tomatoes labels
        with open(label_mapping_path) as f:
            self.tomato_label_ids = {
                row[0] for row in csv.reader(f) if 'tomato' in row[2].lower()
            }

        # Build images list with sampling strategy
        imgs_paths = np.array(self._list_images(imgs_root))
        if sampling == 'under':
            imgs_paths_with_transform = self._undersample(imgs_paths)

        elif sampling == 'over':
            imgs_paths_with_transform = self._oversample(imgs_paths)

        else:
            imgs_paths_with_transform = [(path, None) for path in imgs_paths]

        self.imgs_paths_with_transform = imgs_paths_with_transform

    def __len__(self):
        return len(self.imgs_paths_with_transform)

    def __getitem__(self, idx):
        path, internal_transform = self.imgs_paths_with_transform[idx]
        return self._get_img_tensor(path, internal_transform), self.label_to_vector(self._get_img_label(path))

    def _get_img_tensor(self, fname, internal_transform):
        """Get the corresponding image tensor.
        """
        transforms = list(self.base_transforms)
        if internal_transform:
            transforms.insert(1, internal_transform)

        return T.Compose(transforms)(Image.open(self.imgs_root / fname))

    def _get_img_label(self, path):
        """Get the corresponding image label.
        """
        food_items = self.annotations[path]
        tomato_items = [
            item for item in food_items
            if item['id'] in self.tomato_label_ids
        ]
        return 1 if len(tomato_items) > 0 else 0

    def _undersample(self, imgs_paths):
        """Select the paths with their internal transforms by undersampling.
        """
        labels = [bool(self._get_img_label(path)) for path in imgs_paths]
        without_tomato = imgs_paths[np.invert(labels)]
        with_tomato = imgs_paths[labels]

        indices = np.random.choice(
            with_tomato.shape[0],
            np.sum(labels),
            replace=False
        )

        parts = [imgs_paths[labels], without_tomato[indices]]
        imgs_paths = np.concatenate(parts)
        np.random.shuffle(imgs_paths)
        return [(path, None) for path in imgs_paths]

    def _oversample(self, imgs_paths):
        """Select the paths with their internal transforms by oversampling.
        """
        # Split by label
        labels = [bool(self._get_img_label(path)) for path in imgs_paths]
        without_tomato = imgs_paths[np.invert(labels)]
        with_tomato = imgs_paths[labels]

        # Augment the dataset by transformation
        internal_transforms = [
            None,
            T.RandomHorizontalFlip(1),
            T.RandomVerticalFlip(1),
            RotationTransform(-90),
            RotationTransform(90)
        ]

        paths_transforms = [
            (path, transform)
            for path in with_tomato
            for transform in internal_transforms
        ] + [(path, None) for path in without_tomato]

        result = np.array(paths_transforms)
        np.random.shuffle(result)
        return result

    def _list_images(self, root):
        return [
            fname for fname in os.listdir(root)
            if fname.endswith('.jpg') or fname.endswith('.jpeg')
        ]

    @classmethod
    def label_to_vector(cls, label):
        return torch.tensor([float(1-label), float(label)])

    @classmethod
    def prepare_single_image(cls, path):
        """Prepare a single image for processing.
        """
        return T.Compose(cls.base_transforms)(Image.open(path))

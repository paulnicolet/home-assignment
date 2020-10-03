import csv
import json
import os

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class TomatoesDataset(Dataset):
    """Dataset class used to detect presence of tomatoes on images.
    Each item is a tuple of a image tensor and a class ID (0 = tomato).
    """

    def __init__(self, imgs_root, annotations_path, label_mapping_path, sampling=None, seed=42):
        # Make sure to get reproducible results
        np.random.seed(seed)

        self.resize = (600, 600)
        self.imgs_root = imgs_root

        # Read annotations
        with open(annotations_path) as f:
            self.annotations = json.load(f)

        # Extract tomatoes labels
        with open(label_mapping_path) as f:
            self.tomato_label_ids = {
                row[0] for row in csv.reader(f) if 'tomato' in row[2].lower()
            }

        # Build images list
        imgs_paths = np.array(self._list_images(imgs_root))

        if sampling == 'under':
            imgs_paths_transform = self._undersample(imgs_paths)

        if sampling == 'over':
            imgs_paths_transform = self._oversample(imgs_paths)

        self.imgs_paths_transform = imgs_paths_transform

    def __len__(self):
        return len(self.imgs_paths_transform)

    def __getitem__(self, idx):
        path, internal_transform = self.imgs_paths_transform[idx]
        return self._get_img_tensor(path, internal_transform), self._get_img_label(path)

    def _get_img_tensor(self, path, internal_transform):
        transforms = [T.Resize(self.resize), T.ToTensor()]
        if internal_transform:
            transforms.insert(1, internal_transform)

        return T.Compose(transforms)(Image.open(self.imgs_root / path))

    def _get_img_label(self, path):
        food_items = self.annotations[path]
        tomato_items = [
            item for item in food_items
            if item['id'] in self.tomato_label_ids
        ]
        return 1 if len(tomato_items) > 0 else 0

    def _undersample(self, imgs_paths):
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
        labels = [bool(self._get_img_label(path)) for path in imgs_paths]
        without_tomato = imgs_paths[np.invert(labels)]
        with_tomato = imgs_paths[labels]

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
        np.random.shuffle(imgs_paths)
        return result

    def _list_images(self, root):
        return [
            fname for fname in os.listdir(root)
            if fname.endswith('.jpg') or fname.endswith('.jpeg')
        ]

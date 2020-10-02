import csv
import json
import os

import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms as T


class TomatoesDataset(Dataset):
    """Dataset class used to detect presence of tomatoes on images.
    Each item is a tuple of a image tensor and a boolean (True if tomato).
    """

    def __init__(self, imgs_root, annotations_path, label_mapping_path, sampling=None, seed=42):
        # Make sure to get reproducible results
        np.random.seed(seed)

        self.transforms = T.Compose([T.ToTensor()])
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
            imgs_paths = self._undersample(imgs_paths)

        self.imgs_paths = imgs_paths

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        path = self.imgs_paths[idx]
        return self._get_img_tensor(path), self._get_img_label(path)

    def _get_img_tensor(self, path):
        return self.transforms(io.imread(self.imgs_root / path))

    def _get_img_label(self, path):
        food_items = self.annotations[path]
        tomato_items = [
            item for item in food_items
            if item['id'] in self.tomato_label_ids
        ]
        return len(tomato_items) > 0

    def _undersample(self, imgs_paths):
        labels = [self._get_img_label(path) for path in imgs_paths]
        without_tomato = imgs_paths[np.invert(labels)]

        indices = np.random.choice(
            without_tomato.shape[0],
            np.sum(labels),
            replace=False
        )

        parts = [imgs_paths[labels], without_tomato[indices]]
        imgs_paths = np.concatenate(parts)
        np.random.shuffle(imgs_paths)
        return imgs_paths

    def _list_images(self, root):
        return [
            fname for fname in os.listdir(root)
            if fname.endswith('.jpg') or fname.endswith('.jpeg')
        ]

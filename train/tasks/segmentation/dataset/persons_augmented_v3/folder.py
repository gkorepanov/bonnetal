import os
from PIL import Image
from pathlib import Path
import numpy as np


def is_image(path):
    EXTENSIONS = ['.jpg', '.jpeg', '.png']
    return any(path.suffix == ext for ext in EXTENSIONS)


class FolderSegmentationDataset:
    def __init__(self, root_dir: str, ignore_files: list):
        root_dir = Path(root_dir)

        self.filenames_images = sorted([x for x in root_dir.glob('img/*') if is_image(x) and x.name not in ignore_files])
        self.filenames_labels = sorted([x for x in root_dir.glob('lbl/*') if is_image(x) and x.name not in ignore_files])

        assert len(self.filenames_images) == len(self.filenames_labels)

    def __getitem__(self, index):
        filename_image = self.filenames_images[index]
        filename_label = self.filenames_labels[index]

        with open(filename_image, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB'))
        with open(filename_label, 'rb') as f:
            label = (np.array(Image.open(f).convert('L')) / 255).astype(np.uint8)

        return image, label

    def __len__(self):
        return len(self.filenames_images)

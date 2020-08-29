import os
import yaml
import imageio
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset

import matplotlib.pyplot as plt
from PIL import Image

from .coco import COCOPersonDataset
from .augmenters import SegmentationAugmenter


def is_image(filename):
    EXTENSIONS = ['.jpg', '.jpeg', '.png']
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def make_normalizer(means, stds):
    return torchvision.transforms.Normalize(mean=means, std=stds)


def make_inv_normalizer(means, stds):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = 1 / np.array(stds)),
        torchvision.transforms.Normalize(mean = -np.array(means), std = [ 1., 1., 1. ]),
    ])


class AugmentedSegmentationDataset(Dataset):
    def __init__(self, dataset, augmenter, normalizer, is_train: bool):
        super().__init__()
        self.dataset = dataset
        self.augmenter = augmenter
        self.is_train = is_train
        self.tensorize_image = torchvision.transforms.ToTensor()
        self.tensorize_mask = lambda x: torch.from_numpy(np.squeeze(x)).to(torch.float32)
        self.normalizer = normalizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]

        if self.is_train:
            seed = None
        else:
            seed = index

        image, target_mask, prev_mask, prev_image_augmenter, prev_image = self.augmenter.run_augmentations(image, mask, seed)
        image = self.normalizer(self.tensorize_image(image))
        target_mask = self.tensorize_mask(target_mask)

        if prev_mask is not None:
            prev_mask = self.tensorize_mask(prev_mask)
        if prev_image is not None:
            prev_image = self.normalizer(self.tensorize_image(prev_image))

        return image, target_mask, prev_mask, prev_image_augmenter, prev_image


class Parser:
    def __init__(
        self,
        img_prop,
        img_means, img_stds,
        classes,
        train,
        location,
        batch_size,
        workers,
        crop_prop,
        is_gen_prev_mask,
        is_gen_prev_img
    ):
        self.img_prop = img_prop
        self.classes = classes
        self.norm = make_normalizer(img_means, img_stds)
        self.inv_norm = make_inv_normalizer(img_means, img_stds)

        assert len(location) == 1
        augmenter = SegmentationAugmenter(
            scale=(0.5, 2),
            crop_size=(crop_prop['height'], crop_prop['width']),
            is_gen_prev_mask=is_gen_prev_mask,
            is_gen_prev_img=is_gen_prev_img
        )
        coco_val = COCOPersonDataset(root_dir=location[0], is_train=False)
        coco_train = COCOPersonDataset(root_dir=location[0], is_train=True)

        self.train_dataset = AugmentedSegmentationDataset(
            coco_train, augmenter, self.norm, is_train=True)
        self.trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True
        )
        assert len(self.trainloader) > 0

        self.valid_dataset = AugmentedSegmentationDataset(
            coco_val, augmenter, self.norm, is_train=False)
        self.validloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True
        )
        assert len(self.validloader) > 0

    def get_train_set(self):
      return self.trainloader

    def get_valid_set(self):
      return self.validloader

    def get_train_size(self):
      return len(self.trainloader)

    def get_valid_size(self):
      return len(self.validloader)

    def get_img_size(self):
      h = self.img_prop["height"]
      w = self.img_prop["width"]
      d = self.img_prop["depth"]
      return h, w, d

    def get_n_classes(self):
      return len(self.classes)

    def get_inv_normalize(self):
      return self.inv_norm

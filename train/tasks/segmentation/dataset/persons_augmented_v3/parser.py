import os
import yaml
import imageio
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset

import matplotlib.pyplot as plt
from PIL import Image

from tasks.segmentation.dataset.persons_augmented_v3.coco import COCOPersonDataset
from tasks.segmentation.dataset.persons_augmented_v3.augmenters import SegmentationAugmenter


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
        self.nhwc2nchw = lambda x: x.permute(2, 0, 1)
        self.tensorize = lambda x: torch.from_numpy(x.squeeze()).to(torch.float32)
        self.normalizer = normalizer
        self.IMAGE_KEYS = ['prev_image', 'curr_image']
        self.MASK_KEYS = ['prev_mask', 'curr_mask']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]

        if self.is_train:
            seed = None
        else:
            seed = index

        result = self.augmenter.run_augmentations(image, mask, seed)

        for key in result:
            result[key] = self.tensorize(result[key])

        for key in self.IMAGE_KEYS:
            if key in result:
                result[key] = self.normalizer(self.nhwc2nchw(result[key]) / 255)

        for key in self.MASK_KEYS:
            if key in result:
                result[key] = result[key].long()

#        for key in result:
#            print(key, result[key].shape)

        return result


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
        prev_mask_generator=None,
        prev_image_generator=None,
        curr2prev_optical_flow_generator=None
    ):
        self.img_prop = img_prop
        self.classes = classes
        self.norm = make_normalizer(img_means, img_stds)
        self.inv_norm = make_inv_normalizer(img_means, img_stds)

        assert len(location) == 1
        augmenter = SegmentationAugmenter(
            scale=(0.5, 2),
            crop_size=(crop_prop['height'], crop_prop['width']),
            prev_mask_generator=prev_mask_generator,
            prev_image_generator=prev_image_generator,
            curr2prev_optical_flow_generator=curr2prev_optical_flow_generator
        )

        coco_train = COCOPersonDataset(root_dir=location[0], is_train=True)
        self.trainloader = torch.utils.data.DataLoader(
            AugmentedSegmentationDataset(coco_train, augmenter, self.norm, is_train=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True
        )
        assert len(self.trainloader) > 0

        coco_val = COCOPersonDataset(root_dir=location[0], is_train=False)
        self.validloader = torch.utils.data.DataLoader(
            AugmentedSegmentationDataset(coco_val, augmenter, self.norm, is_train=False),
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

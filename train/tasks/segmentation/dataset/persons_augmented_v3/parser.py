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
from tasks.segmentation.dataset.persons_augmented_v3.folder import FolderSegmentationDataset
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


DATASETS = {
    'coco': COCOPersonDataset,
    'folder': FolderSegmentationDataset
}


class Parser:
    def __init__(
        self,
        img_means, img_stds,
        train_datasets, valid_datasets, test_datasets,
        batch_size, num_workers,
        crop_prop,
        prev_mask_generator=None,
        prev_image_generator=None,
        curr2prev_optical_flow_generator=None,
        **kwargs
    ):
        self.norm = make_normalizer(img_means, img_stds)
        self.inv_norm = make_inv_normalizer(img_means, img_stds)

        self.crop_prop = crop_prop
        self.prev_image_generator = prev_image_generator
        self.prev_mask_generator = prev_mask_generator
        self.curr2prev_optical_flow_generator = curr2prev_optical_flow_generator

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainloader = self.make_dataloader(train_datasets, is_train=True)
        self.validloader = self.make_dataloader(valid_datasets, is_train=False)
        self.testloader = self.make_dataloader(test_datasets, is_train=False)

    def make_dataloader(self, params, is_train):
        datasets = []
        for param in params:
            scale = tuple(param['scale']) if 'scale' in param else None
            augmenter = SegmentationAugmenter(
                scale=scale,
                crop_size=(self.crop_prop['height'], self.crop_prop['width']),
                baseline_augmenter=param['baseline_augmenter'],
                prev_mask_generator=self.prev_mask_generator,
                prev_image_generator=self.prev_image_generator,
                curr2prev_optical_flow_generator=self.curr2prev_optical_flow_generator
            )
            extra_params = param['extra'] if 'extra' in param else dict()
            dataset = DATASETS[param['name']](root_dir=param['location'], **extra_params)
            augmented_dataset = AugmentedSegmentationDataset(dataset, augmenter, self.norm, is_train=False)
            datasets.append(augmented_dataset)

        dataset = ConcatDataset(datasets)
        return torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def get_inv_normalize(self):
      return self.inv_norm

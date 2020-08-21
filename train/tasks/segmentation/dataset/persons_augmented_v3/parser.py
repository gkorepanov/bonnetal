import os
import yaml
import imageio
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import matplotlib.pyplot as plt
from PIL import Image
import cv2



def is_image(filename):
    EXTENSIONS = ['.jpg', '.jpeg', '.png']
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def make_image_augmenter(scale, crop_size):
    h, w = crop_size
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale={'x': (0.95, 1.05), 'y': 1}, fit_output=True),
        iaa.Affine(scale=scale, fit_output=True),
        iaa.size.CropToFixedSize(w, h),
        iaa.size.PadToFixedSize(w, h),
        iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7))),
        iaa.Sometimes(0.1, iaa.OneOf([
            iaa.Sequential([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=True),
                iaa.GaussianBlur(sigma=(0, 1)),
            ])
        ])),
    ], random_order=False)


def make_input_mask_augmenter():
    def dilate(segmaps, random_state, parents, hooks):
        result = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        for segmap in segmaps:
            arr = cv2.morphologyEx(segmap.arr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            result.append(SegmentationMapsOnImage(arr, shape=arr.shape))
        return result

    def drop(segmaps, random_state, parents, hooks):
        return [SegmentationMapsOnImage(np.zeros_like(x.arr), shape=x.arr.shape) for x in segmaps]

    return iaa.Sequential([
        iaa.Lambda(func_segmentation_maps=dilate),
        iaa.Sometimes(0.1,
            iaa.Lambda(func_segmentation_maps=drop), # no mask at all
            iaa.Sometimes(0.1,
                [  # failure case
                    iaa.PiecewiseAffine(scale=(0.02, 0.1), nb_rows=(2, 4), nb_cols=(2, 4)),
                    iaa.ElasticTransformation(alpha=(0, 70), sigma=(4, 8))
                ],
                [  # normal case
                    iaa.PiecewiseAffine(scale=iap.Absolute(iap.TruncatedNormal(0, 0.01, low=-0.04, high=0.04)), nb_rows=(2, 4), nb_cols=(2, 4)),
                    iaa.ElasticTransformation(alpha=(0, 4), sigma=(4, 8))
                ]
            )
        )
    ])


class SegmentationAugmenter:
    def __init__(self, scale, crop_size, is_augment_input_mask: bool):
        super().__init__()
        self.image_augmenter = make_image_augmenter(scale=scale, crop_size=crop_size)
        if is_augment_input_mask:
            self.input_mask_augmenter = make_input_mask_augmenter()
        self.is_augment_input_mask = is_augment_input_mask

    def to_imgaug_format(self, image, label):
        image = np.array(image)
        segmap = SegmentationMapsOnImage(label, image.shape)
        return image, segmap

    def run_augmentations(self, image, label, seed=None):
        image, segmap = self.to_imgaug_format(image, label)
        self.init_random_state(seed)

        image, segmap = self.image_augmenter(image=image, segmentation_maps=segmap)
        output_segmap = segmap.arr

        if self.is_augment_input_mask:
            input_segmap = self.input_mask_augmenter(segmentation_maps=segmap).arr
        else:
            input_segmap = None

        return image, output_segmap, input_segmap

    @staticmethod
    def init_random_state(seed=None):
      if seed is None:
          np.random.seed()
          ia.seed(np.random.get_state()[1])
      else:
          np.random.seed(seed)
          ia.seed(seed)


class COCOPersonDataset:
    def __init__(self, root_dir: str, is_train: bool):
        subset = 'train2017' if is_train else 'val_2017'

        from pycocotools.coco import COCO
        self.images_directory = f'{root_dir}/{subset}'
        self.annotations_file = f'{root_dir}/annotations/{subset}.json'
        self.coco = COCO(annotations_file)
        self.filter_classes = self.coco.getCatIds(catNms=['person'])
        self.image_ids = self.coco.getImgIds(catIds=self.filter_classes)

    def __getitem__(self, index):
        coco_img = self.coco.loadImgs(self.image_ids[index])[0]
        image = imageio.imread(f"{self.images_directory} / {coco_img['file_name']}")
        annotations_ids = self.coco.getAnnIds(imgIds=coco_img['id'], catIds=self.filter_classes, iscrowd=None)
        coco_annotations = coco.loadAnns(annotations_ids)
        filterClasses = ['person']
        mask = np.zeros((coco_img['height'], coco_img['width']))
        for annotation in coco_annotations:
            if annotation['category_id'] not in self.filter_classes:
                continue
            mask[coco.annToMask(annotation)] = 1
        return image, mask, abs(hash(coco_img['file_name']))

    def __len__(self):
        return len(self.image_ids)


def make_normalizer(means, stds):
    return torchvision.transforms.Normalize(mean=means, std=stds)


def make_inv_normalizer(means, stds):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = 1 / np.array(self.img_stds)),
        torchvision.transforms.Normalize(mean = -np.array(self.img_means), std = [ 1., 1., 1. ]),
    ])


class AugmenetedSegmentationDataset(Dataset):
    def __init__(self, dataset, augmenter, normalizer, is_train: bool):
        super().__init__()
        self.dataset = dataset
        self.augmenter = augmenter
        self.is_train = is_train
        self.tensorize_image = torchvision.transforms.ToTensor()
        self.tensorize_mask = lambda x: torch.from_numpy(np.squeeze(x)).long()
        self.normalizer = normalizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask, seed = self.dataset[index]

        if self.is_train:
            seed = None

        image, target_mask, input_mask = self.augmenter.run_augmentations(image, mask, seed)
        image = self.normalizer(self.tensorize_image(image))
        target_mask = self.tensorize_mask(target_mask)

        if input_mask is not None:
            input = torch.cat([
                image,
                self.tensorize_mask(input_mask).unsqueeze(0).unsqueeze(0)
            ], dim=1)
        else:
            input = image

        return input, target_mask


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
        crop_prop
    ):
        self.img_prop = img_prop
        self.classes = classes
        self.norm = make_normalizer(img_means, img_stds)
        self.inv_norm = make_inv_normalizer(img_means, img_stds)

        assert len(location) == 1
        augmenter = SegmentationAugmenter(
            scale=(0.5, 2),
            crop_size=(crop_prop['height'], crop_prop['width']),
            is_augment_input_mask=True
        )
        coco_val = COCOPersonDataset(root_dir=location[0], is_train=False)
        coco_train = COCOPersonDataset(root_dir=location[0], is_train=True)

        self.train_dataset = AugmenetedSegmentationDataset(
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

        self.valid_dataset = AugmenetedSegmentationDataset(
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

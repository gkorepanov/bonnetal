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
        iaa.Sometimes(0.2, iaa.MotionBlur(k=(3, 7))),
        iaa.Sometimes(0.1, iaa.OneOf([
            iaa.Sequential([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=True),
                iaa.GaussianBlur(sigma=(0, 1)),
            ])
        ])),
    ], random_order=False)


def make_prev_augmenter(crop_size):
    h, w = crop_size

    def choose_random_objects(segmaps, random_state, parents, hooks):
        result = []
        for segmap in segmaps:
            mask = segmap.arr
            classes = [x for x in np.unique(mask) if x != 1]
            num_classes_to_choose = np.random.randint(min(1, len(classes)), len(classes) + 1)
            classes_to_choose = list(np.random.choice(classes, num_classes_to_choose, replace=False))
            if np.random.random() > 0.1:
                classes_to_choose.append(1)
            # print(classes, num_classes_to_choose, classes_to_choose)
            mask = np.isin(mask, classes_to_choose)
            result.append(SegmentationMapsOnImage(mask.astype(np.uint8), shape=mask.shape))
        return result


    # def total_dropout(segmaps, random_state, parents, hooks):
    #     result = []
    #     for segmap in segmaps:
    #         mask = segmap.arr
    #         if np.random.random() > 0.5:
    #             mask = np.zeros_like(mask)
    #         else:
    #             mask = np.ones_like(mask)
    #         result.append(SegmentationMapsOnImage(mask, shape=mask.shape))
    #     return result

    def morph_close(segmaps, random_state, parents, hooks):
        result = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for segmap in segmaps:
            arr = cv2.morphologyEx(segmap.arr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            result.append(SegmentationMapsOnImage(arr, shape=arr.shape))
        return result
    
    def make_morph_operation(operation, min_coef=0, max_coef=0.2):
        def f(segmaps, random_state, parents, hooks):
            result = []
            for segmap in segmaps:
                mask = segmap.arr
                indices = np.where(np.any(mask, axis=1))[0]
                if len(indices) == 0:
                    result.append(segmap)
                    continue
                h = indices[-1] - indices[0]
                indices = np.where(np.any(mask, axis=0))[0]
                if len(indices) == 0:
                    result.append(segmap)
                    continue
                w = indices[-1] - indices[0]

                size = min(h, w)
                low = max(2, int(size * min_coef))
                high = max(low + 1, int(size * max_coef))
                kernel_size = np.random.randint(low, high)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = operation(mask.astype(np.uint8), kernel)
                result.append(SegmentationMapsOnImage(mask, shape=mask.shape))
            return result
        return f

    return iaa.Sequential([
        iaa.Sometimes(0.25, iaa.Lambda(func_segmentation_maps=choose_random_objects)),
        iaa.Lambda(func_segmentation_maps=morph_close),
        iaa.Sometimes(0.2,
            # failed mask
            iaa.OneOf([
                iaa.TotalDropout(1.0),  # fill image
                iaa.Sequential([  # fail mask
                    iaa.OneOf([
                        iaa.Lambda(func_segmentation_maps=make_morph_operation(cv2.erode, min_coef=0.2, max_coef=0.5)),
                        iaa.Lambda(func_segmentation_maps=make_morph_operation(cv2.dilate, min_coef=0.2, max_coef=0.5)),
                    ]),
                    iaa.Affine(translate_percent=iap.Choice([iap.Uniform(-0.5, -0.2), iap.Uniform(0.2, 0.5)]))
                ])
            ]),

            # normal mask
            iaa.Sequential([
                iaa.Sometimes(0.1, iaa.OneOf([
                    iaa.Lambda(func_segmentation_maps=make_morph_operation(cv2.erode)),  # smaller mask
                    iaa.Lambda(func_segmentation_maps=make_morph_operation(cv2.dilate)),  # larger mask
                ])),
                iaa.Sometimes(1.0, iaa.Affine(
                    scale=iap.Normal(loc=1, scale=0.02),
                    translate_percent=iap.Normal(loc=0, scale=0.03),
                    shear=iap.Normal(loc=0, scale=1),
                    backend='cv2'
                )),
                iaa.Sometimes(0.1,
                    iaa.ElasticTransformation(alpha=2000, sigma=50)
                ),
                iaa.Sometimes(0.1,
                    iaa.PiecewiseAffine()
                )
            ])
        )
    ], random_order=False)


class SegmentationAugmenter:
    def __init__(self, scale, crop_size, is_gen_prev_mask: bool, is_gen_prev_img: bool):
        super().__init__()
        h, w = crop_size
        self.image_augmenter = make_image_augmenter(scale=scale, crop_size=crop_size)
        self.prev_augmenter = make_prev_augmenter(crop_size=crop_size)
        self.is_gen_prev_mask = is_gen_prev_mask
        self.pad = iaa.size.PadToFixedSize(w, h)
        def mask_postprocess(x):
            x = (x == 1).astype(np.uint8)
            if x.shape[:2] == crop_size: return x
            return cv2.resize(x, tuple(reversed(crop_size)), interpolation=cv2.INTER_LINEAR)
        self.mask_postprocess = mask_postprocess

    def to_imgaug_format(self, image, label):
        image = np.array(image)
        label = np.array(label, dtype=np.uint8)
        segmap = SegmentationMapsOnImage(label, image.shape)
        return image, segmap

    def run_augmentations(self, image, label, seed=None):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image[..., :3]
        image, segmap = self.to_imgaug_format(image, label)
        self.init_random_state(seed)
        pad = self.pad.to_deterministic()

        image, segmap = self.image_augmenter(image=image, segmentation_maps=segmap)
        image, output_segmap = pad(image=image, segmentation_maps=segmap)
        output_segmap = self.mask_postprocess(output_segmap.arr)

        if self.is_gen_prev_mask:
            input_segmap = self.prev_augmenter(segmentation_maps=segmap)
            input_segmap = self.mask_postprocess(pad(segmentation_maps=input_segmap).arr)
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
        subset = 'train2017' if is_train else 'val2017'

        from pycocotools.coco import COCO
        self.images_directory = f'{root_dir}/{subset}'
        self.annotations_file = f'{root_dir}/annotations/instances_{subset}.json'
        self.coco = COCO(self.annotations_file)
        self.filter_classes = self.coco.getCatIds(catNms=['person'])
        self.image_ids = self.coco.getImgIds(catIds=self.filter_classes)

    def __getitem__(self, index):
        coco_img = self.coco.loadImgs(self.image_ids[index])[0]
        image = imageio.imread(f"{self.images_directory}/{coco_img['file_name']}")
        annotations_ids = self.coco.getAnnIds(imgIds=coco_img['id'], catIds=[], iscrowd=None)
        coco_annotations = self.coco.loadAnns(annotations_ids)
        mask = np.zeros((coco_img['height'], coco_img['width']))
        for annotation in coco_annotations:
            mask[self.coco.annToMask(annotation).astype(bool)] = annotation['category_id']
        return image, mask

    def __len__(self):
        return len(self.image_ids)


def make_normalizer(means, stds):
    return torchvision.transforms.Normalize(mean=means, std=stds)


def make_inv_normalizer(means, stds):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = 1 / np.array(stds)),
        torchvision.transforms.Normalize(mean = -np.array(means), std = [ 1., 1., 1. ]),
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
        image, mask = self.dataset[index]

        if self.is_train:
            seed = None
        else:
            seed = index

        image, target_mask, prev_mask = self.augmenter.run_augmentations(image, mask, seed)
        image = self.normalizer(self.tensorize_image(image))
        target_mask = self.tensorize_mask(target_mask)

        if prev_mask is not None:
            prev_mask = self.tensorize_mask(prev_mask)
            input = torch.cat([
                image,
                prev_mask.unsqueeze(0)
            ], dim=0)
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
        batch_size=1,
        workers=1,
        crop_prop=None
    ):
        self.img_prop = img_prop
        self.classes = classes
        self.norm = make_normalizer(img_means, img_stds)
        self.inv_norm = make_inv_normalizer(img_means, img_stds)

        assert len(location) == 1
        augmenter = SegmentationAugmenter(
            scale=(0.5, 2),
            crop_size=(crop_prop['height'], crop_prop['width']),
            is_gen_prev_mask=True,
            is_gen_prev_img=False
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

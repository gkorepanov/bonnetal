# This file is covered by the LICENSE file in the root of this project.

import os
import yaml
import imageio
import numpy as np

import torch
import torchvision

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


def is_image(filename):
    EXTENSIONS = ['.jpg', '.jpeg', '.png']
    return any(filename.endswith(ext) for ext in EXTENSIONS)


class SegmentationDataset(Dataset):
    def __init__(self, root, subset, h, w, means, stds):
        self.images_root = os.path.join(root, subset, "img")
        self.labels_root = os.path.join(root, subset, "lbl")

        self.subset = subset
        assert self.subset == 'train' or self.subset == 'valid'

        self.w = w
        self.h = h
        self.means = means
        self.stds = stds

        print("Images from: ", self.images_root)
        print("Labels from: ", self.labels_root)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
        self.filenamesGt.sort()

        assert len(self.filenames) == len(self.filenamesGt)
        
        self.tensorize_image = torchvision.transforms.ToTensor()
        self.tensorize_label = lambda x: torch.from_numpy(np.squeeze(x)).long()
        self.norm = torchvision.transforms.Normalize(mean=self.means, std=self.stds)
        self.inv_norm = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = 1 / np.array(self.stds)),
            torchvision.transforms.Normalize(mean = -np.array(self.means), std = [ 1., 1., 1. ]),
        ])
        
    def run_augmentations(self, image, label):
        return self.to_imgaug_format(image, label)
    
    def normalize_and_tensorize(self, image, label):
        image = self.tensorize_image(image)
        label = self.tensorize_label(label)
        image = self.norm(image)
        return image, label

    def to_imgaug_format(self, image, label):
        image = np.array(image)
        segmap = SegmentationMapsOnImage((np.array(label) / 255).astype(bool), image.shape)
        return image, segmap

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = Image.open(f).convert('L')

        return self.run_augmentations(image, label)

    def __len__(self):
        return len(self.filenames)


class Persons(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(256/144),
            iaa.Fliplr(0.5),
            iaa.Affine(scale={'x': (0.9, 1.1), 'y': 1}),
            iaa.Affine(rotate=iap.Normal(0, 10), scale=(0.5, 2), translate_percent={'x': 0, 'y': iap.Normal(0, 0.12)}),
            iaa.Resize({'height': 144, 'width': 256}, interpolation='linear'),
            iaa.Sometimes(0.2, iaa.OneOf([
                iaa.Sequential([
                    iaa.pillike.EnhanceBrightness((0.2, 0.9)),
                    iaa.LinearContrast((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                    iaa.GaussianBlur(sigma=(0, 1)),
                ]),
                iaa.pillike.EnhanceBrightness((1.1, 1.6))
            ])),
        ], random_order=False)


    def run_augmentations(self, image, label):
        image, segmap = self.to_imgaug_format(image, label)
        image, segmap = self.augmentations(image=image, segmentation_maps=segmap)
        return self.normalize_and_tensorize(image, segmap.arr)


class Parser():
  # standard conv, BN, relu
  def __init__(self, img_prop, img_means, img_stds, classes, train, location=None, batch_size=None, workers=2):
    super(Parser, self).__init__()

    self.img_prop = img_prop
    self.img_means = img_means
    self.img_stds = img_stds
    self.classes = classes
    self.train = train

    if self.train:
      # if I am training, get the dataset
      self.location = location
      self.batch_size = batch_size
      self.workers = workers

      def make_train_dataset(loc):
        return Persons(root=loc,
                       subset='train',
                       h=self.img_prop["height"],
                       w=self.img_prop["width"],
                       means=self.img_means,
                       stds=self.img_stds)

      # Data loading code
      if isinstance(self.location, list) and len(self.location) > 1:
        self.train_dataset = ConcatDataset([make_train_dataset(loc) for loc in self.location])
      else:
        self.train_dataset = make_train_dataset(self.location)

      def worker_init_fn(worker_id):
        ia.seed(np.random.get_state()[1][0] + worker_id)

      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True,
                                                     worker_init_fn=worker_init_fn)
      assert len(self.trainloader) > 0
      self.trainiter = iter(self.trainloader)

      # calculate validation batch from train batch and image sizes
      self.val_batch_size = max(1, int(self.batch_size))

      # if gpus are available make val_batch_size at least the number of gpus
      if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        self.val_batch_size = max(
            self.val_batch_size, torch.cuda.device_count())

      print("Inference batch size: ", self.val_batch_size)

      def make_valid_dataset(loc):
        return Persons(root=loc,
                       subset='valid',
                       h=self.img_prop["height"],
                       w=self.img_prop["width"],
                       means=self.img_means,
                       stds=self.img_stds)

      if isinstance(self.location, list) and len(self.location) > 1:
        self.valid_dataset = ConcatDataset([make_valid_dataset(loc) for loc in self.location])
      else:
        self.valid_dataset = make_valid_dataset(self.location)

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                     batch_size=self.val_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True,
                                                     worker_init_fn=worker_init_fn)
      assert len(self.validloader) > 0
      self.validiter = iter(self.validloader)

  def get_train_batch(self):
    images, labels = self.trainiter.next()
    return images, labels

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    images, labels = self.validiter.next()
    return images, labels

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

  def get_class_string(self, idx):
    return self.classes[idx]

  def get_means_stds(self):
    return self.img_means, self.img_stds

  def get_inv_normalize(self):
    return self.valid_dataset.inv_norm

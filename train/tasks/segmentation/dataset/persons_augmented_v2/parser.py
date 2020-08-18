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
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBoxesOnImage, BoundingBox
from PIL import Image
import cv2
from torch.utils.data import Dataset, ConcatDataset


def is_image(filename):
    EXTENSIONS = ['.jpg', '.jpeg', '.png']
    return any(filename.endswith(ext) for ext in EXTENSIONS)


class SegmentationDataset(Dataset):
    def __init__(self, root, subset, h, w, means, stds):
        self.images_root = os.path.join(root, subset, "img")
        self.labels_root = os.path.join(root, subset, "lbl")
        self.faces_root = os.path.join(root, subset, "face")

        self.subset = subset
        assert self.subset == 'train' or self.subset == 'valid'

        self.w = w
        self.h = h
        self.means = means
        self.stds = stds

        print("Images from: ", self.images_root)
        print("Labels from: ", self.labels_root)
        print("Faces from: ", self.faces_root)

        self.filenames_images = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.images_root)) for f in fn if is_image(f)]

        self.filenames_labels = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]

        self.filenames_faces = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(self.faces_root)) for f in fn]

        self.filenames_images.sort()
        self.filenames_labels.sort()
        self.filenames_faces.sort()

        assert len(self.filenames_images) == len(self.filenames_labels) == len(self.filenames_faces)

        self.tensorize_image = torchvision.transforms.ToTensor()
        self.tensorize_label = lambda x: torch.from_numpy(np.squeeze(x)).long()
        self.norm = torchvision.transforms.Normalize(mean=self.means, std=self.stds)

    def run_augmentations(self, image, label, face):
        raise NotImplementedError()

    def normalize_and_tensorize(self, image, label):
        image = self.tensorize_image(image)
        label = self.tensorize_label(label)
        image = self.norm(image)
        return image, label

    def to_imgaug_format(self, image, label, face):
        image = np.array(image)
        bbox = [int(x) for x in face.split(' ')]
        segmap = (np.array(label) / 255).astype(bool)

        segmaps = SegmentationMapsOnImage(segmap, image.shape)
        bboxes = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[3], y1=bbox[0], x2=bbox[1], y2=bbox[2]),
        ], shape=image.shape)
        return image, segmaps, bboxes

    def __getitem__(self, index):
        filename_image = self.filenames_images[index]
        filename_label = self.filenames_labels[index]
        filename_face = self.filenames_faces[index]

        with open(filename_image, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(filename_label, 'rb') as f:
            label = Image.open(f).convert('L')
        with open(filename_face, 'r') as f:
            face = f.read()

        return self.run_augmentations(image, label, face)

    def __len__(self):
        return len(self.filenames_images)


class Persons(SegmentationDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['h'] == 144
        assert kwargs['w'] == 256
        self.augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(scale={'x': (0.95, 1.05), 'y': 1}, rotate=iap.Normal(0, 10)),
            iaa.Sometimes(0.2, iaa.OneOf([
                iaa.Sequential([
                    iaa.pillike.EnhanceBrightness((0.2, 0.9)),
                    iaa.LinearContrast((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                    iaa.GaussianBlur(sigma=(0, 1)),
                ]),
                iaa.pillike.EnhanceBrightness((1.1, 1.6))
            ]))
        ], random_order=False)

    def run_augmentations(self, image, label, face):
        image, segmaps, bboxes = self.to_imgaug_format(image, label, face)
        img, segmaps, bbs = self.augmentations(image=image, segmentation_maps=segmaps, bounding_boxes=bboxes)
        bb = bbs[0]

        max_width = bb.width * 8
        min_width = bb.width * 256 / 144
        average = max_width * 0.65 + min_width * 0.35
        distr = iap.Uniform((min_width, average), (average, max_width))
        width = distr.draw_sample()
        height = width * 144 / 256

        x_min = bb.x2 - width
        x_max = bb.x1

        average = x_min * 0.5 + x_max * 0.5
        distr = iap.Uniform((x_min, average), (average, x_max))
        x = int(distr.draw_sample())

        y_min = bb.y2 - height
        y_max = bb.y1
        average = y_min * 0.3 + y_max * 0.7
        distr = iap.Uniform((y_min, average), (average, y_max))
        y = int(distr.draw_sample())

        x = int(x)
        y = int(y)
        height = int(height)
        width = int(width)

        def augment(img, x, y, height, width):
            img = np.pad(img, (
                (max(0 - y, 0), max(y + height - image.shape[0], 0)),
                (max(0 - x, 0), max(x + width - image.shape[1], 0)),
                (0, 0)
            ))
            x = max(x, 0)
            y = max(y, 0)
            return cv2.resize(img[y:y+height, x:x+width], (256, 144))

        img = augment(img, x, y, height, width)
        segmaps = augment(segmaps.arr.astype(np.uint8), x, y, height, width)
        return self.normalize_and_tensorize(img, segmaps)


class Parser():
  # standard conv, BN, relu
  def __init__(self, img_prop, img_means, img_stds, classes, train, location=None, batch_size=None, workers=2):
    super(Parser, self).__init__()

    self.img_prop = img_prop
    self.img_means = img_means
    self.img_stds = img_stds
    self.classes = classes
    self.train = train
    self.inv_norm = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = 1 / np.array(self.img_stds)),
        torchvision.transforms.Normalize(mean = -np.array(self.img_means), std = [ 1., 1., 1. ]),
    ])

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
      self.train_dataset = ConcatDataset([make_train_dataset(loc) for loc in self.location])

      def worker_init_fn(worker_id):
          np.random.seed()
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

      self.valid_dataset = ConcatDataset([make_valid_dataset(loc) for loc in self.location])

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
    return self.inv_norm

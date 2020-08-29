import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import cv2
import numpy as np


def make_image_augmenter(scale, crop_size):
    """Default augmenter, adds crop and resize"""

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


def choose_random_objects(segmaps, random_state, parents, hooks):
    """Augmenter which randomly adds some extra objects to segmentation mask"""

    result = []
    for segmap in segmaps:
        mask = segmap.arr
        classes = [x for x in np.unique(mask) if x != 1]
        num_classes_to_choose = np.random.randint(min(1, len(classes)), len(classes) + 1)
        classes_to_choose = list(np.random.choice(classes, num_classes_to_choose, replace=False))
        if np.random.random() > 0.1:
            classes_to_choose.append(1)
        mask = np.isin(mask, classes_to_choose)
        result.append(SegmentationMapsOnImage(mask.astype(np.uint8), shape=mask.shape))
    return result


def total_dropout(segmaps, random_state, parents, hooks):
    """Augmenter which randomly either removes mask at all or fills entire image with mask"""

    result = []
    for segmap in segmaps:
        mask = segmap.arr
        if np.random.random() > 0.5:
            mask = np.zeros_like(mask)
        else:
            mask = np.ones_like(mask)
        result.append(SegmentationMapsOnImage(mask, shape=mask.shape))
    return result


def morph_close(segmaps, random_state, parents, hooks):
    """Morphological close augmenter"""

    result = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for segmap in segmaps:
        arr = cv2.morphologyEx(segmap.arr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        result.append(SegmentationMapsOnImage(arr, shape=arr.shape))
    return result


def make_morph_operation(operation, min_coef=0, max_coef=0.2):
    """Generates morhological operation augmenter of segmentation masks

    Args:
        operation: cv2.erode or cv2.dilate
    """

    def f(segmaps, random_state, parents, hooks):
        result = []
        for segmap in segmaps:
            mask = segmap.arr

            if not np.any(mask):
                result.append(segmap)
                continue

            # find mask bbox
            indices = np.where(np.any(mask, axis=1))[0]
            h = indices[-1] - indices[0]
            indices = np.where(np.any(mask, axis=0))[0]
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


def make_prev_mask_augmenter(crop_size):
    h, w = crop_size

    return iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Lambda(func_segmentation_maps=choose_random_objects)),
        iaa.Lambda(func_segmentation_maps=morph_close),
        iaa.Sometimes(0.3,
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


def make_prev_image_augmenter(crop_size):
    h, w = crop_size

    return iaa.Sequential([
        iaa.Affine(
            scale=iap.Normal(loc=1, scale=0.05),
            translate_percent=iap.Normal(loc=0, scale=0.05),
            shear=iap.Normal(loc=0, scale=2),
            backend='cv2'
        ),
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=20, sigma=10)),
        iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7))),
        iaa.Sometimes(0.3, iaa.OneOf([
            iaa.Sequential([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=True),
                iaa.GaussianBlur(sigma=(0, 1)),
            ])
        ])),
    ], random_order=False)


class SegmentationAugmenter:
    def __init__(self, scale, crop_size, is_gen_prev_mask: bool, is_gen_prev_img: bool):
        super().__init__()
        h, w = crop_size
        self.image_augmenter = make_image_augmenter(scale=scale, crop_size=crop_size)
        self.prev_mask_augmenter = make_prev_mask_augmenter(crop_size=crop_size)
        self.prev_image_augmenter = make_prev_image_augmenter(crop_size=crop_size)
        self.is_gen_prev_mask = is_gen_prev_mask
        self.is_gen_prev_img = is_gen_prev_img
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
            input_segmap = self.prev_mask_augmenter(segmentation_maps=segmap)
            input_segmap = self.mask_postprocess(pad(segmentation_maps=input_segmap).arr)
        else:
            input_segmap = None

        if self.is_gen_prev_img:
            prev_image_augmenter = self.prev_image_augmenter.to_deterministic()
            prev_image = prev_image_augmenter(image=image)
        else:
            prev_image_augmenter = None
            prev_image = None

        return image, output_segmap, input_segmap, prev_image_augmenter, prev_image

    @staticmethod
    def init_random_state(seed=None):
      if seed is None:
          np.random.seed()
          ia.seed(np.random.get_state()[1])
      else:
          np.random.seed(seed)
          ia.seed(seed)

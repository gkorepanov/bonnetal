import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import cv2
import numpy as np
import torch


def make_baseline_augmenter_random_crop_spoil(scale, crop_size):
    """Default augmenter, adds crop and resize and random spoils"""

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


def make_baseline_augmenter_resize(crop_size):
    """Augmenter for test set, does only resize"""

    h, w = crop_size
    return iaa.Sequential([
        iaa.size.Resize({'width': w, 'height': h}, interpolation='linear')
    ], random_order=False)


def choose_random_objects_mask_augmenter(segmaps, random_state, parents, hooks):
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


def total_dropout_mask_augmenter(segmaps, random_state, parents, hooks):
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


def morph_close_mask_augmenter(segmaps, random_state, parents, hooks):
    """Morphological close augmenter"""

    result = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for segmap in segmaps:
        arr = cv2.morphologyEx(segmap.arr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        result.append(SegmentationMapsOnImage(arr, shape=arr.shape))
    return result


def make_morph_operation_mask_augmenter(operation, min_coef=0, max_coef=0.2):
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


def make_synthetic_prev_mask_complex_mask_augmenter(crop_size):
    h, w = crop_size

    return iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Lambda(func_segmentation_maps=choose_random_objects_mask_augmenter)),
        iaa.Lambda(func_segmentation_maps=morph_close_mask_augmenter),
        iaa.Sometimes(0.3,
            # failed mask
            iaa.OneOf([
                iaa.TotalDropout(1.0),  # fill image
                iaa.Sequential([  # fail mask
                    iaa.OneOf([
                        iaa.Lambda(func_segmentation_maps=make_morph_operation_mask_augmenter(cv2.erode, min_coef=0.2, max_coef=0.5)),
                        iaa.Lambda(func_segmentation_maps=make_morph_operation_mask_augmenter(cv2.dilate, min_coef=0.2, max_coef=0.5)),
                    ]),
                    iaa.Affine(translate_percent=iap.Choice([iap.Uniform(-0.5, -0.2), iap.Uniform(0.2, 0.5)]))
                ])
            ]),

            # normal mask
            iaa.Sequential([
                iaa.Sometimes(0.1, iaa.OneOf([
                    iaa.Lambda(func_segmentation_maps=make_morph_operation_mask_augmenter(cv2.erode)),  # smaller mask
                    iaa.Lambda(func_segmentation_maps=make_morph_operation_mask_augmenter(cv2.dilate)),  # larger mask
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


def make_local_non_geometric_image_augmenter(crop_size):
    h, w = crop_size

    return iaa.Sequential([
        iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7))),
        iaa.Sometimes(0.3, iaa.OneOf([
            iaa.Sequential([
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=True),
                iaa.GaussianBlur(sigma=(0, 1)),
            ])
        ]))
    ], random_order=False)


def make_optical_flow_augmenter(crop_size):
    h, w = crop_size
    alpha_parameter = iap.Uniform(0, 60)

    def augment(*args, **kwargs):
        alpha = alpha_parameter.draw_sample()
        sigma = alpha / 2.5
        augmenter = iaa.Sequential([
            iaa.Affine(
                scale=iap.Normal(loc=1, scale=0.02),
                translate_percent=iap.Normal(loc=0, scale=0.02),
                shear=iap.Normal(loc=0, scale=2),
                backend='cv2'
            ),
            iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=alpha, sigma=sigma))
        ], random_order=False)
        return augmenter(*args, **kwargs)
    return augment


def make_optical_flow_generator(crop_size):
    h, w = crop_size
    augmenter = make_optical_flow_augmenter(crop_size)
    grid = np.stack(np.meshgrid(np.linspace(1, 3, w), np.linspace(1, 3, h)), axis=-1).astype(np.float32)
    return lambda: augmenter(image=grid) - 2


def apply_optical_flow(image, optical_flow):
    optical_flow_torch = torch.tensor(optical_flow).unsqueeze(0).to(torch.float32)
    image_torch = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    result_torch = torch.nn.functional.grid_sample(image_torch, optical_flow_torch, align_corners=False)
    result = result_torch[0].permute(1, 2, 0).numpy().astype(np.uint8)
    return result


class SegmentationAugmenter:
    def __init__(
        self,
        crop_size,
        scale,
        baseline_augmenter='train',
        prev_mask_generator=None,
        prev_image_generator=None,
        curr2prev_optical_flow_generator=None
    ):
        h, w = crop_size
        if baseline_augmenter == 'random_crop_spoil':
            self.baseline_augmenter = make_baseline_augmenter_random_crop_spoil(scale=scale, crop_size=crop_size)
        elif baseline_generator == 'resize':
            self.baseline_augmenter = make_baseline_augmenter_resize(crop_size=crop_size)
        else:
            raise NotImplementedError()

        self.pad = iaa.size.PadToFixedSize(w, h)

        def mask_postprocess(x):
            """Fix bug with incorrect mask size after imgaug"""
            x = (x == 1).astype(np.uint8)
            if x.shape[:2] == crop_size: return x
            return cv2.resize(x, tuple(reversed(crop_size)), interpolation=cv2.INTER_LINEAR)

        self.mask_postprocess = mask_postprocess

        self.imgaug_affine_elastic_optical_flow_generator = make_optical_flow_generator(crop_size)
        self.local_non_geometric_image_augmenter = make_local_non_geometric_image_augmenter(crop_size)
        self.curr2prev_optical_flow_generator = curr2prev_optical_flow_generator
        self.prev_image_generator = prev_image_generator
        self.prev_mask_generator = prev_mask_generator

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

        image, segmap = self.baseline_augmenter(image=image, segmentation_maps=segmap)
        image, output_segmap = pad(image=image, segmentation_maps=segmap)
        output_segmap = self.mask_postprocess(output_segmap.arr)

        result = {
            'curr_image': image,
            'curr_mask': output_segmap
        }

        if not self.curr2prev_optical_flow_generator:
            pass
        elif self.curr2prev_optical_flow_generator == 'imgaug_affine_elastic':
            result['curr2prev_optical_flow'] = self.imgaug_affine_elastic_optical_flow_generator()
        else:
            raise NotImplementedError()

        if not self.prev_image_generator:
            pass
        elif self.prev_image_generator == 'optical_flow_and_local':
            prev_image = apply_optical_flow(result['curr_image'], result['curr2prev_optical_flow'])
            prev_image = self.local_non_geometric_image_augmenter(image=prev_image)
            result['prev_image'] = prev_image
        else:
            raise NotImplementedError()

        if not self.prev_mask_generator:
            pass
        else:
            raise NotImplementedError()

        return result

    @staticmethod
    def init_random_state(seed=None):
      if seed is None:
          np.random.seed()
          ia.seed(np.random.get_state()[1])
      else:
          np.random.seed(seed)
          ia.seed(seed)

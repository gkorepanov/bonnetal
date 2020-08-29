import numpy as np
import imageio


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

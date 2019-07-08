import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from contextlib import redirect_stdout
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, root, coco_set, resize, max_size, stride, is_training):
        self.root = root
        self.coco_set = coco_set
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.is_training = is_training

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with redirect_stdout(None):     # redirect the output of pycocotools
            self.coco = COCO(os.path.join(root, 'annotations', 'instances_{}.json'.format(self.coco_set)))
        self.image_ids = self.coco.getImgIds()
        self._load_categories()

    def _load_categories(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.category_to_label = {}
        self.label_to_category = {}
        self.label_to_name = {}
        for i, cat in enumerate(categories):
            self.category_to_label[cat['id']] = i
            self.label_to_category[i] = cat['id']
            self.label_to_name[i] = cat['name']

    def _load_image(self, image_id):
        # read image
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_path = os.path.join(self.root, 'images', self.coco_set, image_name)
        image = Image.open(image_path).convert('RGB')
        return image

    def _load_annotations(self, index):
        # get ground truth annotations
        annotation_ids = self.coco.getAnnIds(imgIds=self.image_ids[index], iscrowd=False)
        annotations = self.coco.loadAnns(annotation_ids)

        boxes, categories = [], []
        for ann in annotations:
            # some annotations have basically no width / height, skip them
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            categories.append(self.category_to_label[ann['category_id']])

        if len(boxes) > 0:
            target = (torch.FloatTensor(boxes), torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        if self.is_training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, ids, ratios = zip(*batch)

        # pad data to match max batch dimensions
        sizes = [d.size()[1:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(F.pad(datum, [0, ph, 0, pw]) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.is_training:
            return data, targets

        ids = torch.IntTensor(ids)
        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, ids, ratios

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self._load_image(image_id)

        # randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list) or isinstance(resize, tuple):
            resize = random.randint(self.resize[0], self.resize[1])

        ratio = resize / min(image.size)
        if ratio * max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
        image = image.resize((int(ratio * d) for d in image.size), Image.BILINEAR)

        if self.is_training:
            boxes, categories = self._load_annotations(index)
            boxes *= ratio

            # random horizontal flip
            if random.randint(0, 1):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = image.size[0] - boxes[:, 0] - boxes[:, 2]

            # xywh to xyxy
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] - 1
            target = torch.cat([boxes, categories], dim=1)

        # convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.float().div(255).view(image.size[1], image.size[0], 3)
        data = data.permute(2, 0, 1)

        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        if self.is_training:
            return data, target

        return data, image_id, ratio


class CocoDataLoader(object):
    def __init__(self, root, coco_set, resize, max_size, stride=1, batch_size=16, is_training=False):
        assert coco_set in ['train2014', 'val2014', 'test2014', 'train2017', 'val2017', 'test2017']
        self.resize = resize
        self.max_size = max_size

        self.dataset = CocoDataset(root, coco_set, resize, max_size, stride, is_training)
        self.image_ids = self.dataset.image_ids
        self.coco = self.dataset.coco
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=is_training,
                                 collate_fn=self.dataset.collate_fn, num_workers=2, pin_memory=True)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for output in self.loader:
            if self.dataset.is_training:
                data, target = output
                yield data, target
            else:
                data, ids, ratios = output
                yield data, ids, ratios


if __name__ == '__main__':
    # loader = CocoDataLoader('/root/datasets/COCO/', 'train2014', 200, 240, 1, batch_size=16, is_training=True)
    # data_iter = iter(loader)
    # data, targets = next(data_iter)
    # print(data.size())
    # print(data)
    # print(targets.size())
    # print(targets)

    loader = CocoDataLoader('/root/datasets/COCO/', 'val2014', 200, 240, 1, batch_size=16, is_training=False)
    data_iter = iter(loader)
    data, ids, ratios = next(data_iter)
    print(data.size())
    print(data)
    print(ids.size())
    print(ids)
    print(ratios.size())
    print(ratios)
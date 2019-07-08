import os
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from contextlib import redirect_stdout
from tensorboardX import SummaryWriter
from PIL import Image

from pycocotools.cocoeval import COCOeval

from models.retinanet import RetinaNet
from models.loss import FocalLoss, SmoothL1Loss
from utils.misc import print_cuda_statistics, ImageProcessor
from utils.bbox import *


class Solver(object):
    def __init__(self, config):
        self.config = config

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2**(i/3) for i in range(3)]
        self.anchors = {}

        self.model = RetinaNet(self.config.num_classes, self.config.backbone,
                               self.config.num_features, len(self.ratios) * len(self.scales))

        self.cuda = torch.cuda.is_available() and self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            print("Operation will be on *******GPU******* ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            print("Operation will be on *******CPU******* ")

    def _load_checkpoints(self):
        if self.config.checkpoint is None:
            return
        filename = 'checkpoint_{}.pth'.format(self.config.checkpoint)
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, filename))
        state_dict = checkpoint['state_dict']
        self.iters = self.config.checkpoint
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        if hasattr(self, 'optimizer') and 'optimizer' in state_dict.keys():
            self.optimizer.load_state_dict(checkpoint['state_dict'])

    def _save_checkpoints(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        filename = 'checkpoint_{}.pth'.format(self.iters)
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))

    def _compute_loss(self, data, cls_preds, box_preds, targets):
        """
        Compute classification loss and box regression loss.
        :param data: [N, C, H, W]
        :param cls_preds: [[N, KA, H/S3, W/S3], [N, KA, H/S4, W/S4], ..., [N, KA, H/S7, W/S7]]
        :param box_preds: [[N, 4A, H/S3, W/S3], [N, 4A, H/S4, W/S4], ..., [N, 4A, H/S7, W/S7]]
        :param targets: {N, num_boxes, 5]
        """
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_pred, box_pred in zip(cls_preds, box_preds):
            # cls_pred: [N, KA, H/S, W/S]
            # box_pred: [N, 4A, H/S, W/S]
            size = cls_pred.size()[-2:]
            stride = data.size()[-1] // cls_pred.size()[-1]
            # cls_target: classification target, [NA, K, H/S, W/S]
            # box_target: box regression target, [NA, 4, H/S, W/S]
            # depth: indicates whether a anchor is ignored (-1) or background (0) or object (>0), [NA, 1, H/S, W/S]
            cls_target, box_target, depth = self._assign_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            # calculate classification loss (focal loss)
            cls_pred = cls_pred.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_pred, cls_target)
            cls_loss = cls_mask * cls_loss      # ignore anchors whose overlap is between 0.4 and 0.5
            cls_losses.append(cls_loss.sum())

            # calculate box regression loss (smooth-l1 loss)
            box_pred = box_pred.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_pred, box_target)
            box_loss = box_mask * box_loss      # only consider foreground objects
            box_losses.append(box_loss.sum())

        # normalize losses by the number of anchors assigned to a ground-truth box
        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def _assign_targets(self, targets, stride, size):
        """
        Assign classification target and box target for each anchor on a feature map.
        :param targets: the ground truth of a batch, [N, num_boxes, 5]
        :param stride: the stride of feature map
        :param size: the size of feature map
        :return: classification target, [NA, K, H, W]
                 box regression target, [NA, 4, H, W]
                 depth,                 [NA, 1, H, W]
        """
        cls_target, box_target, depth = [], [], []
        for target in targets:
            # target: ground truth of an image, [num_boxes, 5]
            target = target[target[:, -1] > -1]     # filter padded targets during collating batches
            if stride not in self.anchors.keys():
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            snapped = snap_to_anchors(target,
                                      [s * stride for s in size[::-1]],     # (h, w) to (w, h)
                                      stride,
                                      self.anchors[stride].to(targets.device),
                                      self.config.num_classes,
                                      targets.device)
            for l, s in zip((cls_target, box_target, depth), snapped):
                l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def train(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

        # create tensorboard writer
        self.writer = SummaryWriter(self.config.summary_dir)

        # prepare for optimization
        self.iters = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr,
                                   weight_decay=self.config.weight_decay, momentum=self.config.momentum)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.milestones, gamma=0.1)
        self.cls_criterion = FocalLoss(self.config.focal_alpha, self.config.focal_gamma)
        self.box_criterion = SmoothL1Loss(self.config.smoothl1_beta)

        # prepare model
        self._load_checkpoints()
        self.model.to(self.device)
        if self.cuda and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # training process
        self.model.train()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()

        while self.iters < self.config.max_iters:
            for data, targets in tqdm(self.train_loader):
                # forward
                data, targets = data.to(self.device), targets.to(self.device)
                cls_preds, box_preds = self.model(data)
                cls_loss, box_loss = self._compute_loss(data, cls_preds, box_preds, targets)
                del data, targets

                # backward
                loss = cls_loss + box_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iters += 1
                # summary
                if self.iters % self.config.summary_step == 0:
                    self.writer.add_scalar('cls_loss', cls_loss.item(), self.iters)
                    self.writer.add_scalar('box_loss', box_loss.item(), self.iters)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.iters)
                # save checkpoints
                if self.iters % self.config.save_step == 0:
                    self._save_checkpoints()
                # validation
                if self.val_loader is not None and self.iters % self.config.val_step == 0:
                    self.validate()
                    self.model.train()
                # update learning rate
                self.scheduler.step()

                if self.iters == self.config.max_iters:
                    break

    def validate(self):
        self.model.eval()
        results, image_ids = [], []
        with torch.no_grad():
            for data, ids, ratios in tqdm(self.val_loader):
                data = data.to(self.device)
                cls_preds, box_preds = self.model(data)
                cls_preds = [cls_pred.sigmoid() for cls_pred in cls_preds]

                decoded = []
                for cls_pred, box_pred in zip(cls_preds, box_preds):
                    stride = data.size()[-1] // cls_pred.size()[-1]
                    if stride not in self.anchors:
                        self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
                    # decode and filter boxes
                    decoded.append(decode(cls_pred, box_pred, stride, self.config.dec_threshold, self.config.topn,
                                          self.anchors[stride]))

                # perform non-maximum suppression
                decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
                scores, boxes, classes = nms(*decoded, self.config.nms_threshold, self.config.ndetections)
                scores, boxes, classes = scores.cpu(), boxes.cpu(), classes.cpu()

                # correct boxes for image scales and transform to xywh
                boxes /= ratios.view(-1, 1, 1)
                boxes[:, :, 2:] -= boxes[:, :, :2]

                # iterate over a batch
                for i, (image_scores, image_boxes, image_classes) in enumerate(zip(scores, boxes, classes)):
                    # filter padded entries
                    keep = (image_scores > 0).nonzero().view(-1)
                    image_scores = image_scores.index_select(0, keep)
                    image_boxes = image_boxes.index_select(0, keep)
                    image_classes = image_classes.index_select(0, keep)

                    for image_score, image_box, image_class in zip(image_scores, image_boxes, image_classes):
                        image_result = {
                            'image_id': ids[i].item(),
                            'category_id': self.val_loader.dataset.label_to_category[int(image_class.item())],
                            'score': image_score.item(),
                            'bbox': image_box.view(4).tolist()
                        }
                        results.append(image_result)

                    image_ids.append(ids[i])    # append image to list of processed images

        # write output
        json.dump(results, open('{}/val{}_result.json'.format(self.config.log_dir, self.iters), 'w'), indent=4)

        # run COCO evaluation
        with redirect_stdout(open(os.path.join(self.config.log_dir, 'val_results.txt'), 'w')):
            print('Epoch {}: Validation Results'.format(self.iters))
            # load results in COCO evaluation tool
            coco_true = self.val_loader.dataset.coco
            coco_pred = coco_true.loadRes('{}/val{}_result.json'.format(self.config.log_dir, self.iters))

            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            print()

    def inference(self, image_path):
        # load image
        image = Image.open(image_path)
        processor = ImageProcessor()
        data = processor.normalize(image)
        data = data.to(self.device)

        # prepare model
        self._load_checkpoints()
        self.model.to(self.device)
        self.model.eval()

        # forward
        cls_preds, box_preds = self.model(data.unsqueeze(0))
        cls_preds = [cls_pred.sigmoid() for cls_pred in cls_preds]

        decoded = []
        for cls_pred, box_pred in zip(cls_preds, box_preds):
            stride = data.size()[-1] // cls_pred.size()[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            # decode and filter boxes
            decoded.append(decode(cls_pred, box_pred, stride, self.config.dec_threshold, self.config.topn,
                                  self.anchors[stride]))

        # perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        scores, boxes, classes = nms(*decoded, self.config.nms_threshold, self.config.ndetections)
        scores.squeeze_(0)
        boxes.squeeze_(0)
        classes.squeeze_(0)

        # transform to xywh
        boxes[:, 2:] -= boxes[:, :2]

        # filter padded entries
        keep = (scores > 0).nonzero().view(-1)
        scores = scores.index_select(0, keep).cpu()
        boxes = boxes.index_select(0, keep).cpu()
        classes = classes.index_select(0, keep).cpu()

        detections = []
        for scr, box, cls in zip(scores, boxes, classes):
            detections.append({'score': scr.item(), 'bbox': box.tolist(), 'class': int(cls.item())})

        processor.show_detections(image, detections, self.config.result_dir)
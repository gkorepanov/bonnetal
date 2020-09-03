#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from common.logger import Logger
from backbones.config import *
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.oneshot import OneShot_LR
from tasks.segmentation.modules.head import *
from tasks.segmentation.modules.segmentator import *
from tasks.segmentation.modules.colorizer import *
from tasks.segmentation.modules.ioueval import *


class Trainer():
  def __init__(self, config, logdir, path=None, only_eval=False, block_bn=False):
    # parameters
    self.CFG = config
    self.log = logdir
    self.path = path
    self.block_bn = block_bn

    # put logger where it belongs
    self.tb_logger = Logger(self.log + "/tb")
    self.info = {"train_update": 0,
                 "train_loss": 0,
                 "train_iou": 0,
                 "valid_loss": 0,
                 "valid_iou": 0,
                 "valid_loss_avg_models": 0,
                 "valid_iou_avg_models": 0,
                 "feat_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0}

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/segmentation/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(batch_size=self.CFG["train"]["batch_size"], **self.CFG["dataset"])
    self.training_mode = self.CFG["train"]["training_mode"]

    print(f"TRAIN size: {len(self.parser.trainloader)}")
    print(f"VALID size: {len(self.parser.validloader)}")
    print(f"TEST size: {len(self.parser.testloader)}")

    self.valid_save_image_each = max(1, len(self.parser.validloader) // self.CFG['train']['num_images_to_save'])
    assert self.valid_save_image_each > 0
    print(f"Will save an image each {self.valid_save_image_each} batch during validation")

    self.test_save_image_each = max(1, len(self.parser.testloader) // self.CFG['train']['num_images_to_save'])
    assert self.test_save_image_each > 0
    print(f"Will save an image each {self.test_save_image_each} batch during test")

    self.data_h, self.data_w, self.data_d = self.CFG["train"]["input_size"]

    # weights for loss (and bias)
    self.loss_w = torch.zeros(len(self.CFG["dataset"]["labels"]), dtype=torch.float)
    for idx, w in self.CFG["dataset"]["labels_w"].items():
      self.loss_w[idx] = torch.tensor(w)

    # get architecture and build backbone (with pretrained weights)
    self.bbone_cfg = BackboneConfig(name=self.CFG["backbone"]["name"],
                                    os=self.CFG["backbone"]["OS"],
                                    h=self.data_h,
                                    w=self.data_w,
                                    d=self.data_d,
                                    dropout=self.CFG["backbone"]["dropout"],
                                    bn_d=self.CFG["backbone"]["bn_d"],
                                    extra=self.CFG["backbone"]["extra"])

    self.decoder_cfg = DecoderConfig(name=self.CFG["decoder"]["name"],
                                     dropout=self.CFG["decoder"]["dropout"],
                                     bn_d=self.CFG["decoder"]["bn_d"],
                                     extra=self.CFG["decoder"]["extra"])

    self.head_cfg = HeadConfig(n_class=1,
                               dropout=self.CFG["head"]["dropout"])

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.bbone_cfg,
                               self.decoder_cfg,
                               self.head_cfg,
                               self.path)

    # train backbone?
    if not self.CFG["backbone"]["train"]:
      self.CFG["backbone"]["train"] = False
      for w in self.model.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.CFG["decoder"]["train"]:
      self.CFG["decoder"]["train"] = False
      for w in self.model.decoder.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel()
                        for p in self.model.parameters())
    weights_grad = sum(p.numel()
                       for p in self.model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)
    # breakdown by layer
    weights_enc = sum(p.numel()
                      for p in self.model.backbone.parameters())
    weights_dec = sum(p.numel()
                      for p in self.model.decoder.parameters())
    weights_head = sum(p.numel()
                       for p in self.model.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)

    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.n_gpus = 0
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      self.gpu = True
      # cudnn.benchmark = True
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)   # spread in gpus
      # self.model = convert_model(self.model).cuda()  # sync batchnorm
      self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True
      self.n_gpus = torch.cuda.device_count()

    # loss
    if "loss" in self.CFG["train"].keys() and self.CFG["train"]["loss"] == "xentropy":
      criterion = nn.CrossEntropyLoss(weight=self.loss_w)
    elif "loss" in self.CFG["train"].keys() and self.CFG["train"]["loss"] == "iou":
      criterion = mIoULoss(weight=self.loss_w)
    elif self.CFG["train"].get("loss") == "xentropy_dice":
      criterion = CrossEntropyDiceLoss(weight=self.loss_w)
    else:
      raise Exception('Loss not defined in config file')

    self.criterion = criterion.to(self.device)
    self.temporal_criterion = nn.MSELoss().to(self.device)

    # loss as dataparallel too (more images in batch)
    if self.n_gpus > 1:
      self.criterion = nn.DataParallel(self.criterion).cuda()
      self.temporal_criterion = nn.DataParallel(self.temporal_criterion).cuda()

    # optimizer
    train_dicts = [{'params': self.model_single.head.parameters()}]
    if self.CFG["backbone"]["train"]:
      train_dicts.append({'params': self.model_single.backbone.parameters()})
    if self.CFG["decoder"]["train"]:
      train_dicts.append({'params': self.model_single.decoder.parameters()})

    # Use SGD optimizer to train
    self.optimizer = optim.SGD(train_dicts,
                               lr=self.CFG["train"]["max_lr"],
                               momentum=self.CFG["train"]["min_momentum"],
                               weight_decay=self.CFG["train"]["w_decay"])

    # Use one shot learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = len(self.parser.trainloader)
    up_steps = int(self.CFG["train"]["up_epochs"] * steps_per_epoch)
    down_steps = int(self.CFG["train"]["down_epochs"] * steps_per_epoch)
    final_decay = self.CFG["train"]["final_decay"] ** (1/steps_per_epoch)

    if self.CFG["train"]["scheduler"] == 'SGDR':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,10 * steps_per_epoch, T_mult=1.5)
    elif self.CFG["train"]["scheduler"] == 'oneshot':
      self.scheduler = OneShot_LR(self.optimizer,
                                  base_lr=self.CFG["train"]["min_lr"],
                                  max_lr=self.CFG["train"]["max_lr"],
                                  step_size_up=up_steps,
                                  step_size_down=down_steps,
                                  cycle_momentum=True,
                                  base_momentum=self.CFG["train"]["min_momentum"],
                                  max_momentum=self.CFG["train"]["max_momentum"],
                                  post_decay=final_decay)
    else:
      raise NotImplementedError()

    if self.CFG["train"].get("last_epoch"):
        for _ in range(self.CFG["train"]["last_epoch"]): self.scheduler.step()

    # buffer to save the best N models
    self.best_n_models = self.CFG["train"]["avg_N"]
    self.best_backbones = collections.deque(maxlen=self.best_n_models)
    self.best_decoders = collections.deque(maxlen=self.best_n_models)
    self.best_heads = collections.deque(maxlen=self.best_n_models)

  def save_checkpoint(self, bbone, decoder, head, suffix=""):
    # Save the weights
    torch.save(bbone, self.log + "/backbone" + suffix)
    torch.save(decoder, self.log + "/segmentation_decoder" + suffix)
    torch.save(head, self.log + "/segmentation_head" + suffix)

  def save_to_log(self, logdir, logger, info, epoch, w_summary=False, rand_imgs=dict(), img_summary=False):
    # save scalars
    for tag, value in info.items():
      logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
    if w_summary:
      for tag, value in self.model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
          logger.histo_summary(
              tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    if img_summary:
      for tag, imgs in rand_imgs.items():
        logger.image_summary(tag, imgs, epoch)

  def train(self):
    self.best_val_iou = 0.0

    self.ignore_class = []
    for i, w in enumerate(self.loss_w):
      if w < 1e-10:
        self.ignore_class.append(i)
        print("Ignoring class ", i, " in IoU evaluation")

    self.evaluator = iouEval(len(self.CFG["dataset"]["labels"]),
                             self.device, self.ignore_class)

    # image colorizer
    self.colorizer = Colorizer(self.CFG["dataset"]["color_map"])

    # train for n epochs
    for epoch in range(self.CFG["train"]["max_epochs"]):
      # get info for learn rate currently

      groups = self.optimizer.param_groups
      if len(groups) == 3:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = groups[1]['lr']
        self.info["feat_lr"] = groups[2]['lr']
      elif len(groups) == 2:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = groups[1]['lr']
        self.info["feat_lr"] = 0
      elif len(groups) == 1:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = 0
        self.info["feat_lr"] = 0
      else:
        print("Invalid learning rate groups optimizer")

      if epoch % self.CFG["train"]["report_epoch"] == 0:
        self.report_validation(epoch=epoch)

      # train for 1 epoch
      iou, losses, update_mean = self.train_epoch(train_loader=self.parser.trainloader,
                                                     model=self.model,
                                                     optimizer=self.optimizer,
                                                     epoch=epoch,
                                                     evaluator=self.evaluator,
                                                     block_bn=self.block_bn,
                                                     scheduler=self.scheduler)

      # update info
      for key in losses:
        self.info['train_' + key] = losses[key]

      self.info["train_update"] = update_mean
      # self.info["train_iou"] = iou

    print('Finished Training')

    return

  def report_validation(self, epoch):
    # evaluate on validation set
    print("*" * 100)
    print("Validation on valid set")
    images = dict()
    iou, losses, rand_img = self.validate(
      val_loader=self.parser.validloader,
      model=self.model,
      evaluator=self.evaluator,
      save_images=self.CFG["train"]["save_imgs"],
      class_dict=self.CFG["dataset"]["labels"],
      save_image_each=self.valid_save_image_each
    )

    images['valid'] = rand_img

    # update info
    for key in losses:
        self.info['valid_' + key] = losses[key]

    self.info["valid_iou"] = iou

    # remember best iou and save checkpoint
    if iou > self.best_val_iou and epoch >= 0:
      print("Best mean iou in validation so far, save model!")
      print("*" * 100)
      self.best_val_iou = iou

      # save the weights!
      current_backbone = self.model_single.backbone.state_dict()
      current_decoder = self.model_single.decoder.state_dict()
      current_head = self.model_single.head.state_dict()
      self.save_checkpoint(
          current_backbone, current_decoder, current_head, suffix="")

      if False:
        self.average_models_and_save()

    print("*" * 100)
    print("Validation on test set")
    iou, losses, rand_img = self.validate(
      val_loader=self.parser.testloader,
      model=self.model,
      evaluator=self.evaluator,
      save_images=self.CFG["train"]["save_imgs"],
      class_dict=self.CFG["dataset"]["labels"],
      save_image_each=self.test_save_image_each
    )

    images['test'] = rand_img

    # update info
    for key in losses:
        self.info['test_' + key] = losses[key]

    self.info["test_iou"] = iou

    # save to log
    self.save_to_log(logdir=self.log,
                      logger=self.tb_logger,
                      info=self.info,
                      epoch=epoch,
                      w_summary=self.CFG["train"]["save_summary"],
                      rand_imgs=images,
                      img_summary=self.CFG["train"]["save_imgs"])
    print("*" * 100)

  def average_models_and_save(self):
    raise NotImplementedError()
    # now average the models and evaluate again
    print("Averaging the best {0} models".format(self.best_n_models))

    # append current backbone to its circular buffer
    current_backbone = self.model_single.backbone.state_dict()
    avg_backbone = copy.deepcopy(
        self.model_single.backbone).cpu().state_dict()
    self.best_backbones.append(copy.deepcopy(
        self.model_single.backbone).cpu().state_dict())

    # now average the backbone
    for i, backbone in enumerate(self.best_backbones):
        # for each weight key
      for key, val in backbone.items():
        # if it is the first time, zero the entry first
        if i == 0:
          avg_backbone[key].data.zero_()
        # then sum the avg contribution
        avg_backbone[key] += (backbone[key] / \
            float(len(self.best_backbones))).to(avg_backbone[key].dtype)

    # append current backbone to its circular buffer
    current_decoder = self.model_single.decoder.state_dict()
    avg_decoder = copy.deepcopy(
        self.model_single.decoder).cpu().state_dict()
    self.best_decoders.append(copy.deepcopy(
        self.model_single.decoder).cpu().state_dict())

    # now average the decoder
    for i, decoder in enumerate(self.best_decoders):
      # for each weight key
      for key, val in decoder.items():
        # if it is the first time, zero the entry first
        if i == 0:
          avg_decoder[key].data.zero_()
        # then sum the avg contribution
        avg_decoder[key] += (decoder[key] / \
            float(len(self.best_decoders))).to(decoder[key].dtype)

    # append current head to its circular buffer
    current_head = self.model_single.head.state_dict()
    avg_head = copy.deepcopy(self.model_single.head).cpu().state_dict()
    self.best_heads.append(copy.deepcopy(
        self.model_single.head).cpu().state_dict())

    # now average the head
    for i, head in enumerate(self.best_heads):
      # for each weight key
      for key, val in head.items():
        # if it is the first time, zero the entry first
        if i == 0:
          avg_head[key].data.zero_()
        # then sum the avg contribution
        avg_head[key] += (head[key] / float(len(self.best_heads))).to(head[key].dtype)

    # put averaged weights in dictionary and evaluate again
    self.model_single.backbone.load_state_dict(avg_backbone)
    self.model_single.decoder.load_state_dict(avg_decoder)
    self.model_single.head.load_state_dict(avg_head)

    # evaluate on validation set
    iou, losses, _ = self.validate(val_loader=self.parser.validloader,
                                      model=self.model,
                                      evaluator=self.evaluator,
                                      save_images=self.CFG["train"]["save_imgs"],
                                      class_dict=self.CFG["dataset"]["labels"])

    # update info
    for key in losses:
      self.info['valid_' + key + '_avg_models'] = losses[key]
    self.info["valid_iou_avg_models"] = iou

    # restore the current weights into model
    self.model_single.backbone.load_state_dict(current_backbone)
    self.model_single.decoder.load_state_dict(current_decoder)
    self.model_single.head.load_state_dict(current_head)

    # save the weights!
    self.save_checkpoint(
        current_backbone, current_decoder, current_head, suffix="_single")
    self.save_checkpoint(avg_backbone, avg_decoder, avg_head, suffix="_average")

  def run_single_batch(self, model, batch):
      if self.gpu:
        for key in batch:
          batch[key] = batch[key].cuda()

      if self.training_mode == 'temporal_loss':
        curr_output = model(image=batch['curr_image'], mask=None)
        gt_loss = self.criterion(torch.cat([1 - curr_output, curr_output], dim=1), batch['curr_mask'])

        prev_output = model(image=batch['prev_image'], mask=None)
        pseudo_prev_mask = F.grid_sample(curr_output, batch['curr2prev_optical_flow'])
        pseudo_prev_mask = pseudo_prev_mask.detach()  # previously this was not done and hence might work worse
        temporal_loss = self.temporal_criterion(prev_output, pseudo_prev_mask)
        loss = gt_loss + self.CFG["train"]["temporal_loss_strength"] * temporal_loss

        batch_result = {
          'losses': {
            'gt_loss': gt_loss.mean().item(),
            'temporal_loss': temporal_loss.mean().item(),
            'loss': loss.mean().item(),
          },
          'curr_output': curr_output,
          'prev_output': prev_output,
          'pseudo_prev_mask': pseudo_prev_mask
        }
      elif self.training_mode == 'temporal_loss_mask_prop':
        n, c, h, w = batch['curr_image'].shape
        empty_mask = torch.zeros(size=(n, 1, h, w),  dtype=torch.float32, requires_grad=False)
        if self.gpu:
          empty_mask = empty_mask.cuda()
        curr_output = model(image=batch['curr_image'], mask=empty_mask)
        gt_loss = self.criterion(torch.cat([1 - curr_output, curr_output], dim=1), batch['curr_mask'])

        prev_output = model(image=batch['prev_image'], mask=curr_output.detach())
        pseudo_prev_mask = F.grid_sample(curr_output, batch['curr2prev_optical_flow'])
        pseudo_prev_mask = pseudo_prev_mask.detach()
        temporal_loss = self.temporal_criterion(prev_output, pseudo_prev_mask)
        prev_gt_loss = self.criterion(torch.cat([1 - prev_output, prev_output], dim=1), batch['prev_mask'])
        loss = (
          gt_loss +
          self.CFG["train"]["temporal_loss_strength"] * temporal_loss +
          self.CFG["train"]["prev_gt_loss_strength"] * prev_gt_loss
        )

        batch_result = {
          'losses': {
            'gt_loss': gt_loss.mean().item(),
            'temporal_loss': temporal_loss.mean().item(),
            'loss': loss.mean().item(),
            'prev_gt_loss': prev_gt_loss.mean().item()
          },
          'curr_output': curr_output,
          'prev_output': prev_output,
          'pseudo_prev_mask': pseudo_prev_mask
        }
      else:
        raise NotImplementedError()

      return loss, batch_result

  def train_epoch(self, train_loader, model, optimizer, epoch, evaluator, block_bn, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()
    losses = collections.defaultdict(AverageMeter)

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    # switch batchnorm to eval mode if I want to block rolling averages
    if block_bn:
      for m in model.modules():
        if isinstance(m, nn.modules.BatchNorm1d):
          m.eval()
        if isinstance(m, nn.modules.BatchNorm2d):
          m.eval()
        if isinstance(m, nn.modules.BatchNorm3d):
          m.eval()

    loss_idx = torch.ones(self.n_gpus).cuda() if self.n_gpus > 1 else None

    end = time.time()
    for i, batch in enumerate(train_loader):
      batch_size = batch['curr_image'].size(0)
      # measure data loading time
      data_time.update(time.time() - end)

      loss, batch_result = self.run_single_batch(model=model, batch=batch)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward(loss_idx)
      optimizer.step()

      # record loss
      for key in batch_result['losses']:
        losses[key].update(batch_result['losses'][key], batch_size)

      # with torch.no_grad():
        # evaluator.reset()
        # evaluator.addBatch(output.argmax(dim=1), target)
        # jaccard, class_jaccard = evaluator.getIoU()
      # iou.update(class_jaccard[-1].item(), batch_size)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so to print the relationship of their norms
      lr = self.optimizer.param_groups[0]["lr"]
      update_ratios = []
      for _, value in self.model.named_parameters():
        if value.grad is not None:
          w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
          update = np.linalg.norm(-lr * value.grad.cpu().numpy().reshape((-1)))
          update_ratios.append(update / w)
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch

      if i % self.CFG["train"]["report_batch"] == 0:
        print('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses['loss'], iou=iou, lr=lr,
                  umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()

    return iou.avg, {k: v.avg for k, v in losses.items()}, update_ratio_meter.avg

  def binarize_output_mask(self, output_mask):
    return (output_mask > 0.5).long()

  def validate(self, val_loader, model, evaluator, save_images, class_dict, save_image_each):
    batch_time = AverageMeter()
    losses = collections.defaultdict(AverageMeter)
    rand_imgs = []

    # switch to evaluate mode
    model.eval()
    evaluator.reset()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, batch in enumerate(val_loader):
        batch_size = batch['curr_image'].size(0)
        _, batch_result = self.run_single_batch(model=model, batch=batch)

        # measure accuracy and record loss
        evaluator.addBatch(self.binarize_output_mask(batch_result['curr_output']), batch['curr_mask'])

        # record loss
        for key in batch_result['losses']:
          losses[key].update(batch_result['losses'][key], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # save a random image, if desired
        if save_images and (i % save_image_each == 0):
          index = np.random.randint(0, batch_size - 1)
          rand_imgs.append(self.make_log_image(batch=batch, batch_result=batch_result, index=index))

      jaccard, class_jaccard = evaluator.getIoU()
      iou = class_jaccard[1].item()

      print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'IoU avg {iou:.3f}'.format(batch_time=batch_time,
                                           loss=losses['loss'], iou=iou))
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_dict[i], jacc=jacc))

    return iou, {k: v.avg for k, v in losses.items()}, rand_imgs

  def make_log_image(self, batch, batch_result, index):
      # for key, tensor in batch.items():
      #   print(key, tensor.shape)
      # for key, tensor in batch_result.items():
      #   print(key, tensor.shape)

      curr_image = self.parser.get_inv_normalize()(batch['curr_image'][index])
      curr_image = curr_image.permute(1, 2, 0).cpu().numpy() * 255
      sep = np.ones((curr_image.shape[0], 2, 3)) * 255

      curr_mask = batch['curr_mask'][index].cpu().numpy()
      curr_output = self.binarize_output_mask(batch_result['curr_output'][index]).squeeze().cpu().numpy()

      result = [
        curr_image,
        self.colorizer.do(curr_mask) * 0.5 + curr_image * 0.5,
        self.colorizer.do(curr_output) * 0.5 + curr_image * 0.5
      ]

      if self.training_mode in ('temporal_loss', 'temporal_loss_mask_prop'):
          prev_image = self.parser.get_inv_normalize()(batch['prev_image'][index])
          prev_image = prev_image.permute(1, 2, 0).cpu().numpy() * 255

          prev_output = self.binarize_output_mask(batch_result['prev_output'][index]).squeeze().cpu().numpy()
          pseudo_prev_mask = self.binarize_output_mask(batch_result['pseudo_prev_mask'][index]).squeeze().cpu().numpy()

          result += [
            self.colorizer.do(prev_output) * 0.5 + prev_image * 0.5,
            self.colorizer.do(prev_output - pseudo_prev_mask) * 0.5 + prev_image * 0.5
          ]
      else:
          raise NotImplementedError()

      # for i, tensor in enumerate(result):
      #     print(i, tensor.shape)

      return np.concatenate(sum([[x, sep] for x in result], [])[:-1], axis=1).astype(np.uint8)

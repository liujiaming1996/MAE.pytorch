import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import utils.utils as utils
from einops import rearrange
from timm.data import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import ModelEma, accuracy


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)[0][-1][-1]
    outputs = model.module.classifier(outputs)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def pretrain_one_epoch(model: torch.nn.Module,
                       data_loader: Iterable,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device,
                       epoch: int,
                       loss_scaler,
                       max_norm: float = 0,
                       patch_size: int = 16,
                       normlize_target: bool = True,
                       log_writer=None,
                       lr_scheduler=None,
                       start_steps=None,
                       lr_schedule_values=None,
                       wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for step, (images, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for _, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images = images.to(device, non_blocking=True)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN, device=device)
            std = torch.as_tensor(IMAGENET_DEFAULT_STD, device=device)
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
            unnorm_images = images * std + mean  # re-norm into [0, 1]

            if normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (images_squeeze.var(
                        dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size,
                    p2=patch_size)

            # B, _, C = images_patch.shape
            # labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            loss = model(images, images_patch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss,
                                optimizer,
                                clip_grad=max_norm,
                                parameters=model.parameters(),
                                create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def finetune_one_epoch(model: torch.nn.Module,
                       criterion: torch.nn.Module,
                       data_loader: Iterable,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device,
                       epoch: int,
                       loss_scaler,
                       max_norm: float = 0,
                       model_ema: Optional[ModelEma] = None,
                       mixup_fn: Optional[Mixup] = None,
                       log_writer=None,
                       start_steps=None,
                       lr_schedule_values=None,
                       wd_schedule_values=None,
                       num_training_steps_per_epoch=None,
                       update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if 'lr_scale' in param_group.keys():
                        param_group["lr"] = lr_schedule_values[it] * param_group[
                            "lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it] * param_group[
                            "lr"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets,
                                             criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(model, samples, targets,
                                                 criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss,
                                    optimizer,
                                    clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) %
                                    update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            loss, output = train_class_batch(model, images, target,
                                             criterion)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

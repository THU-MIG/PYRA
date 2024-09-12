import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

from model.tome import apply_tome

# flops count
from fvcore.nn import FlopCountAnalysis

def test_model_latency(model, device, test_batch_size=512):
    T0 = 10
    T1 = 10
    speed = 0
    model.eval()
    with torch.no_grad():
        x = torch.randn(test_batch_size, 3, 224, 224).to(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < T0:   
            model(x)
        torch.cuda.synchronize()
        print("*****Test model latency (images per second)*****")
        timing = []
        while sum(timing) < T1:
            start = time.time()
            model(x)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
        timing = torch.as_tensor(timing, dtype=torch.float32)
        speed=512/timing.mean().item()
        print("Model latency: {} imgs/s".format(speed))
    
    return speed
    # exit(0)

def count_model_flops(model, device):
    model.eval()
    rand_input = torch.rand((1,3,224,224)).to(device)
    flops = FlopCountAnalysis(model,rand_input)
    print("*****Count model FLOPS (GFLOPS)*****")
    total_flops = flops.total()
    print("Total FLOPS:%d, GFLOPS:%.4f"%(total_flops, float(total_flops)/1073741824))
    print(flops.by_module_and_operator())
    return float(total_flops)/1073741824

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None,
                    deit=False):

    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:

            if not deit:
                outputs = model(samples)
            else:
                outputs, _ = model(samples)

            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        grad_clip = False
        if not math.isfinite(loss_value):
            print("Loss is {}, clipping gradient".format(loss_value))
            grad_clip = True

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 10)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

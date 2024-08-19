""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from timm.optim.adabelief import AdaBelief
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lamb import Lamb
from timm.optim.lars import Lars
from timm.optim.lookahead import Lookahead
from timm.optim.madgrad import MADGRAD
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def pre_process(model, weight_decay=1e-5, skip_list=(), separate_lr_for_pyra=False, pyra_lr=0):
    decay = []
    no_decay = []
    ssf = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            if separate_lr_for_pyra and "ssf" in name:
                ssf.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': ssf, 'weight_decay': weight_decay, 'lr':pyra_lr}]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
        separate_lr_for_pyra=args.separate_lr_for_pyra,
        pyra_lr=args.pyra_lr
    )


def create_optimizer_v2(
        model_or_params,
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        separate_lr_for_pyra: bool = False,
        pyra_lr: float = 0,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        if (weight_decay and filter_bias_and_bn) or separate_lr_for_pyra:
            skip = {}
            if hasattr(model_or_params, 'no_weight_decay'):
                skip = model_or_params.no_weight_decay()
            parameters = pre_process(model_or_params, weight_decay, skip, separate_lr_for_pyra, pyra_lr)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = optim.Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

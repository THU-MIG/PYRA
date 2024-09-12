import os
import cv2
import json
import torch
import scipy
import scipy.io as sio
from skimage import io

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

class general_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None,**kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))

def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    if args.data_set == 'clevr_count':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 8
    elif args.data_set == 'diabetic_retinopathy':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 5
    elif args.data_set == 'dsprites_loc':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == 'dtd':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 47
    elif args.data_set == 'kitti':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 4
    elif args.data_set == 'oxford_pet':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == 'resisc45':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 45
    elif args.data_set == 'smallnorb_ele':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 9
    elif args.data_set == 'svhn':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'cifar100':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'clevr_dist':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == 'caltech101':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'dmlab':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == 'dsprites_ori':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == 'eurosat':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'oxford_flowers102':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == 'patch_camelyon':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 2
    elif args.data_set == 'smallnorb_azi':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 18
    elif args.data_set == 'sun397':
        dataset = general_dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 397

    return dataset, nb_classes

def build_transform(is_train, args):
    if not args.no_aug and is_train and args.mode != 'search':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        return transform

    t = []
    if args.direct_resize:
        size = args.input_size
    else:
        size = int((256 / 224) * args.input_size)

    t.append(
        transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.inception:
        t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

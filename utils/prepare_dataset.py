import torch
import torch.utils.data
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

from utils.tiny_imagenet import TinyImageNetDataset
from utils.tiny_imagenet_c import TinyImageNetCDataset
from utils.visdatest import *

NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tinyimagenet_transforms = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

visda_train = transforms.Compose([transforms.Resize((256,256)),
                                  transforms.RandomCrop((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

visda_val = transforms.Compose([transforms.Resize((256,256)),
                                transforms.CenterCrop((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
n_px = 224
def _convert_image_to_rgb(image):
    return image.convert("RGB")
clip_transforms = transforms.Compose([transforms.Resize(n_px, interpolation=BICUBIC),
                                      transforms.CenterCrop(n_px),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

augment_transforms = transforms.Compose([transforms.RandomRotation(180),
                                         transforms.ColorJitter()])


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_test_data(args, transform=None):
    if args.clip :
        te_transforms = clip_transforms
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
            teset.data = teset_raw

        elif args.corruption == 'cifar_new':
            from utils.cifar_new import CIFAR_New
            teset = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/', transform=te_transforms)
            permute = False
        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=False, transform=te_transforms)

            teset.data = teset_raw
    elif args.dataset == 'visda':
        teset = VisdaTest(args.dataroot, transforms=visda_val)

    elif args.dataset == 'tiny-imagenet':
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = TinyImageNetDataset(args.dataroot + '/tiny-imagenet-200/', mode='val', transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset = TinyImageNetCDataset(args.dataroot + '/Tiny-ImageNet-C/', corruption = args.corruption, level = args.level,
                                        transform=te_transforms)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        te_sampler = torch.utils.data.distributed.DistributedSampler(teset)
    else:
        te_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    if args.distributed:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
            shuffle=(te_sampler is None), num_workers=args.workers, pin_memory=True, sampler=te_sampler)
    else:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)

    return teloader, te_sampler, teset


def prepare_val_data(args, transform=None):
    if args.dataset == 'visda':
        vset = ImageFolder(root=args.dataroot + 'validation/', transform=transform if transform is not None else visda_val)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        v_sampler = torch.utils.data.distributed.DistributedSampler(vset)
    else:
        v_sampler = None
    if not hasattr(args, 'workers'):
        args.workers = 1
    vloader = torch.utils.data.DataLoader(vset, batch_size=args.batch_size,
        shuffle=(v_sampler is None), num_workers=args.workers, pin_memory=True, sampler=v_sampler, drop_last=True)
    return vloader, v_sampler

def prepare_train_data(args, transform=None):
    if args.clip :
        tr_transforms = clip_transforms
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=True, download=False, transform=tr_transforms)
    elif args.dataset == 'cifar100':
        trset = torchvision.datasets.CIFAR100(root=args.dataroot, train=True, download=False, transform=tr_transforms)
    elif args.dataset == 'visda':
        dataset = ImageFolder(root=args.dataroot + 'train/', transform=visda_train if transform is None else transform)
        trset, _ = random_split(dataset, [106678, 45719], generator=torch.Generator().manual_seed(args.seed))
    elif args.dataset == 'tiny-imagenet':
        trset = TinyImageNetDataset(args.dataroot + '/tiny-imagenet-200/', transform=tinyimagenet_transforms)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(trset)
    else:
        tr_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
        shuffle=(tr_sampler is None), num_workers=args.workers, pin_memory=True, sampler=tr_sampler)


    return trloader, tr_sampler, trset


def prepare_image_folder_data(args, transform=None):

    if args.clip :
        transform = clip_transforms

    
    set = ImageFolder(root=args.dataroot, transform=transform)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(set)
    else:
        sampler = None
    if not hasattr(args, 'workers'):
        args.workers = 1
    loader = torch.utils.data.DataLoader(set, batch_size=args.batch_size,
        shuffle=(sampler is None), num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)
    return loader, sampler, set
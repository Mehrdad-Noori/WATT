import argparse

def argparser():
    parser = argparse.ArgumentParser()

    #Directories
    parser.add_argument('--root', type=str, default='/home/davidoso/scratch/Projects/MaskUpProj/', help='Base path')
    parser.add_argument('--dataroot', type=str, default='/home/davidoso/Documents/Data/')
    parser.add_argument('--save', type=str, default='work/', help='Path fo r base training weights')
    parser.add_argument('--save-iter', type=str, default='work/', help='Path for base training weights')

    #General settings
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=10, help='Number of epochs to print progress')
    parser.add_argument('--clip', type=bool, default=True)

    #Model
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--load', type=str)

    #Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'tiny-imagenet', 'imagenet', 'visda', 'PACS', 'office_home', 'VLCS'))
    parser.add_argument('--target', type=str, default='cifar10')
    parser.add_argument('--workers', type=int, default=6, help='Number of workers for dataloader')

    #Source training
    parser.add_argument('--epochs', type=int, default=100, help='Number of base training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Manual epoch number for restarts')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for base training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay',  type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--evaluate', action='store_true', help='Evaluating on evaluation set')
    parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
    parser.add_argument('--method', type=str, default='parallel', choices=('parallel', 'sequential'))
    parser.add_argument('--validate-text-features', action='store_true')
    parser.add_argument('--exps', default=3, type=int, help = 'number of experiments to create (mean +/- std)')

    #Test-Time Adaptation
    parser.add_argument('--adapt', action='store_true', help='To adapt or not')
    parser.add_argument('--level', default=5, type=int)
    parser.add_argument('--corruption', default='gaussian_noise')
    parser.add_argument('--adapt-lr', default=0.00001, type=float)
    parser.add_argument('--niter', default=10, type=int)

    #Distributed
    parser.add_argument('--distributed', action='store_true', help='Activate distributed training')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:3456', help='url for distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
    parser.add_argument('--world-size', type=int, default=1, help='Number of nodes for training')

    return parser.parse_args()

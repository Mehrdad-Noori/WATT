import os
import gdown
import argparse
import tarfile
from zipfile import ZipFile
from torchvision.datasets import CIFAR10, CIFAR100


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


# CIFAR10 ########################################################################
def download_cifar10(data_dir):
    full_path = stage_path(data_dir, "CIFAR10")

    CIFAR10(root=full_path, download=True)



# CIFAR-10.1 ########################################################################
def download_cifar10_1(data_dir):
    # Original URL: https://github.com/modestyachts/CIFAR-10.1
    
    download_and_extract(
        "https://drive.google.com/uc?id=1cdc5OEyLlWJSkfHEUBUlmjYQF2FSkjUL",
        os.path.join(data_dir, "CIFAR10", "CIFAR-10.1.zip"),
    )



# CIFAR-10-C ########################################################################
def download_cifar10c(data_dir):
    # Original URL: https://zenodo.org/records/2535967#.YzHFMXbMJPY
    
    download_and_extract(
        "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1",
        os.path.join(data_dir, "CIFAR10", "CIFAR-10-C.tar"),
    )



# CIFAR100 ########################################################################
def download_cifar100(data_dir):
    full_path = stage_path(data_dir, "CIFAR100")

    CIFAR100(root=full_path, download=True)


# CIFAR-10-C0 ########################################################################
def download_cifar100c(data_dir):
    # Original URL: https://zenodo.org/records/2535967#.YzHFMXbMJPY
    
    download_and_extract(
        "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
        os.path.join(data_dir, "CIFAR100", "CIFAR-100-C.tar"),
    )


# VLCS ########################################################################
def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")

    download_and_extract(
        "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
        os.path.join(data_dir, "VLCS.tar.gz"),
    )


# PACS ########################################################################
def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract(
        "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
        os.path.join(data_dir, "PACS.zip"),
    )

    os.rename(os.path.join(data_dir, "kfold"), full_path)


# Office-Home #################################################################
def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract(
        "https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
        os.path.join(data_dir, "office_home.zip"),
    )

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"), full_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    download_cifar10(args.data_dir)
    download_cifar10_1(args.data_dir)
    download_cifar10c(args.data_dir)
    download_cifar100(args.data_dir)
    download_cifar100c(args.data_dir)
    download_vlcs(args.data_dir)
    download_pacs(args.data_dir)
    download_office_home(args.data_dir)

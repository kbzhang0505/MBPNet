
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import os
import os.path


class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


def get_transform(opt, grayscale=False, method=InterpolationMode.BICUBIC, convert=True):#Make the resize and Crop tools
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    if convert:
        transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


#Check whether it is a file
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


#Create data sets based on paths
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]#Returns a linked list of image paths of a specified size within a folder

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.base_dataset import make_dataset
from PIL import Image
import torchvision.transforms as transforms

import numpy as np

class RandomCrop(object):#Cropped picture randomly, size 0.8

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['LI'], sample['HI']

        h, w = image.shape[:2]
        min_a = min(h, w)
        self.output_size = (min_a * 8 // 10, min_a * 8 // 10)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w]

        landmarks = landmarks[top: top + new_h,left: left + new_w]

        return {'LI': image, 'HI': landmarks}


#Initialize the whole dataset
class OursDataset(BaseDataset):

    def __init__(self, opt):
        # Instantiate the underlying data set
        BaseDataset.__init__(self, opt)
        if opt.phase == 'train':
            #Fill the full path
            self.train_dir_HI = os.path.join(opt.dataroot, 'trainhigh')#label
            self.train_dir_LI = os.path.join(opt.dataroot, 'trainlow')#Input

            #Get all image paths list
            self.train_HI_paths = sorted(make_dataset(self.train_dir_HI, opt.max_dataset_size))
            self.train_LI_paths = sorted(make_dataset(self.train_dir_LI, opt.max_dataset_size))
            self.train_size = len(self.train_HI_paths)  # Dataset size

            self.crop=RandomCrop(opt.load_size)
        else:#Test
            self.test_dir_HI = os.path.join(opt.dataroot, 'testhigh')#label
            self.test_dir_LI = os.path.join(opt.dataroot, 'testlow')#Input
            #Get all image paths list
            self.test_HI_paths = sorted(make_dataset(self.test_dir_HI, opt.max_dataset_size))
            self.test_LI_paths = sorted(make_dataset(self.test_dir_LI, opt.max_dataset_size))
            self.test_size = len(self.test_LI_paths)  # get the size of dataset

        #Make the resize tool
        input_nc =self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.output_nc  # get the number of channels of output image
        self.transform_HI = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_LI = get_transform(self.opt, grayscale=(output_nc == 1))
        print(self.transform_HI)

        if 'resize' in self.opt.preprocess:
            int12=int(opt.load_size / 2)
            int14=int(opt.load_size / 4)
            osize12 = [int12, int12]
            osize14 = [int14, int14]
            self.trans2 = transforms.Compose([transforms.Resize(osize12), transforms.ToTensor()])
            self.trans4 = transforms.Compose([transforms.Resize(osize14), transforms.ToTensor()])

    # Get a batchsize of data randomly cropped and converted to the specified size
    def __getitem__(self, index):

        if self.opt.phase == 'train':
            train_index = index % self.train_size# make sure index is within then range
            HI_path = self.train_HI_paths[train_index]
            LI_path = self.train_LI_paths[train_index]

            HI_img = np.asarray(Image.open(HI_path).convert('RGB'))
            LI_img = np.asarray(Image.open(LI_path).convert('RGB'))
            imgs = self.crop({'LI': LI_img, 'HI': HI_img})
            HI_img, LI_img = Image.fromarray(imgs['HI']), Image.fromarray(imgs['LI'])

        else:  # test
            test_index = index % self.test_size

            LI_path = self.test_LI_paths[test_index]
            LI_img = Image.open(LI_path).convert('RGB')
            if test_index < len(self.test_HI_paths):
                HI_path = self.test_HI_paths[test_index]
                HI_img = Image.open(HI_path).convert('RGB')
            else:
                HI_img = Image.fromarray(np.zeros_like(LI_img))

        w, h = HI_img.size
        neww = w // 8 * 8
        newh = h // 8 * 8
        resize = transforms.Resize([newh, neww])
        HI_img = resize(HI_img)
        LI_img = resize(LI_img)

        HI = self.transform_HI(HI_img)
        LI = self.transform_LI(LI_img)

        if 'resize' in self.opt.preprocess:
            HI2 = self.trans2(HI_img)
            HI4 = self.trans4(HI_img)
        else:
            self.trans2 = transforms.Compose([transforms.Resize([int(newh/2), int(neww/2)]), transforms.ToTensor()])
            self.trans4 = transforms.Compose([transforms.Resize([int(newh/4), int(neww/4)]), transforms.ToTensor()])
            HI2 = self.trans2(HI_img)
            HI4 = self.trans4(HI_img)

        return {'HI': HI, 'HI2': HI2, 'HI4': HI4, 'LI': LI, 'LI_paths': LI_path}

    def __len__(self):

        if self.opt.dataset_size == 0 or self.opt.phase == 'test':
            length = self.test_size
        else:
            length = self.opt.dataset_size
        return length

    def getlen(self):
        if self.opt.dataset_size == 0 or self.opt.phase == 'test':
            length = self.test_size
        else:
            length = self.opt.dataset_size
        return length


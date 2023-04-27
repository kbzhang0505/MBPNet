import argparse
import os
from util import util
import torch


class BaseOptions:

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--dataroot', type=str, default='./datasets/training_data', help='Dataset path')
        parser.add_argument('--name', type=str, default='MBPN(MIT)', help='Name of experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids:0,1,2,use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--preprocess', type=str, default='resize', help='Uniform picture size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')

        parser.add_argument('--input_nc', type=int, default=3, help='input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# in the first conv layer')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='')

        parser.add_argument('--serial_batches', action='store_true', help='takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        parser.add_argument('--phase', type=str, default='train', help='train,test')
        parser.add_argument('--dataset_size', type=int, default=4850, help='train dataset size')
        parser.add_argument('--maxepoch', type=int, default=500, help='Maximum training epoch')

        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--if_savetraini', action='store_true', help='Whether to keep the middle training picture')

        parser.add_argument('--lr', type=float, default=0.0004, help='learning rate for adam')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--isTrain', type=bool, default=True, help='Whether the training')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective')

        parser.set_defaults(no_dropout=False)

        self.initialized = True
        return parser

    def parse(self):#Tectonic return opt

        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)

        self.parser = parser

        opt = parser.parse_args()

        self.print_options(opt)

        # set gpu
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt

        return self.opt

    # Print run options and save options as files
    def print_options(self, opt):

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
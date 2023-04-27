
import os
import ntpath
import time
from util import util


def save_images(visuals, image_path,image_dir):

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_datas in visuals.items():
        if not isinstance(im_datas, list):
            im_datas = [im_datas]
        for i, im_data in enumerate(im_datas):
            im = util.tensor2im(im_data)
            image_name = '%s_%s_%02d.png' % (name, label, i)
            save_path = os.path.join(image_dir, image_name)

            util.save_image(im, save_path)


class Visualizer():

    def __init__(self, opt):

        self.opt = opt
        self.use_traini = opt.isTrain and (not opt.if_savetraini)
        self.name = opt.name
        self.saved = False

        if self.use_traini:
            self.traini_dir = os.path.join(opt.checkpoints_dir, opt.name, 'traini')
            self.img_dir = os.path.join(self.traini_dir, 'images')
            print('create web directory %s...' % self.traini_dir)
            util.mkdirs([self.traini_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def display_current_results(self, visuals, iters):

        if self.use_traini and (not self.saved):
            self.saved = True
            # save images to the disk
            for label, im_datas in visuals.items():
                if not isinstance(im_datas, list):
                    im_datas = [im_datas]
                for i, im_data in enumerate(im_datas):
                    image_numpy = util.tensor2im(im_data)
                    img_path = os.path.join(self.img_dir, 'iters%.3d_%s_%d.png' % (iters, label, i))
                    util.save_image(image_numpy, img_path)


    def print_current_losses(self, epoch, iters, losses, t_comp):

        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_comp)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


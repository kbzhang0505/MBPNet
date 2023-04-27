
import os
from util.visualizer import save_images
from util import util
from data import create_dataset
import copy
from models.ours_model import MBPNModel


class envclass():

    def __init__(self, opt):

        self.testopt = copy.deepcopy(opt)

        self.testopt.num_threads = 1
        self.testopt.batch_size = 1
        self.testopt.phase = "test"
        self.testopt.preprocess = ""
        self.testopt.results_dir = './results/'
        self.testopt.isTrain = False

        self.testdataset = create_dataset(self.testopt)

        self.bestpsnr = 0.0

    def env(self, epoch):
        temppsnr=""
        self.testopt.epoch = str(epoch)
        self.testmodel = MBPNModel(self.testopt)
        self.testmodel.setup(self.testopt)
        print("testmodel:%s"%str(epoch))

        for i, data in enumerate(self.testdataset):
            self.testmodel.set_input(data)
            self.testmodel.test()

        testdata = self.testmodel.gettest()
        print("testdata:" + str(self.testdataset.dataset.getlen()) + "/" + str(testdata))

        if testdata['PSNR']>self.bestpsnr:
            self.bestpsnr = testdata['PSNR']
            temppsnr = "bestmode"

        log_name = os.path.join(self.testopt.checkpoints_dir, self.testopt.name, 'envdata_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s:%s_%s\n' % (str(epoch),testdata,temppsnr))
        return temppsnr

    def envsave(self, epoch):

        self.testopt.epoch = str(epoch)
        self.img_dir = os.path.join(self.testopt.results_dir, self.testopt.name, '%s_%s' % (self.testopt.phase, self.testopt.epoch), 'images')
        util.mkdirs(self.img_dir)

        self.testmodel = MBPNModel(self.testopt)
        self.testmodel.setup(self.testopt)
        print("testmodel:")

        for i, data in enumerate(self.testdataset):

            self.testmodel.set_input(data)
            self.testmodel.test()

            visuals = self.testmodel.get_current_visuals()
            img_path = self.testmodel.get_image_paths()

            print('processing (%04d)-th image... %s' % (i, img_path))

            save_images(visuals, img_path,self.img_dir)

        testdata = self.testmodel.gettest()
        print("testdata:" + str(self.testdataset.dataset.getlen()) + "/" + str(testdata))

        log_name = os.path.join(self.img_dir, 'testmodel_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % testdata)




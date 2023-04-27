import torch
import itertools
from util.testdata import AverageMeters
from util.util import tensor2im
from .base_model import BaseModel
from . import networks
from . import vgg
import numpy as np
import util.index as index
import lpips

class MBPNModel(BaseModel, torch.nn.Module):

    def __init__(self, opt):

        BaseModel.__init__(self, opt)#Initialize the base model
        torch.nn.Module.__init__(self)
        self.loss_names = ['Pix', 'MP', 'Adv', 'Grad', 'Color', 'T']

        self.visual_names = ['boost_HIS', 'real_HI']

        if self.isTrain:
            self.model_names = ['G', 'D']

        else:
            self.model_names = ['G']
            self.avg_meters = AverageMeters()
            self.loss_fn = lpips.LPIPS(net='alex', version='0.1')

        # Define generator
        self.netG = networks.define_G(opt.input_nc * 3, opt.input_nc, opt.ngf, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)

            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.25)#Prevent gradient is too large, clipping

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionGra = networks.GradientLoss(self.device)  # Gradient loss
            self.criterionColor = networks.ColorLoss()  # Color loss
            self.criterionVgg = networks.VGGLoss1(self.device, vgg=self.vgg, normalize=False)#VGG loss
            self.criterionmse = torch.nn.MSELoss()#Mean-square loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)#Antagonistic loss

        self.real_LI = None
        self.real_HI = None
        self.real_HI2 = None
        self.real_HI4 = None


    def set_input(self, input):#Puts input data into the specified variable for use

        with torch.no_grad():
            if self.isTrain:
                self.real_HI2 = input['HI2'].to(self.device)
                self.real_HI4 = input['HI4'].to(self.device)

                #Image pairs
                LI = input['LI']
                HI = input['HI']
            else:  # Test
                self.LI_paths = input['LI_paths']
                LI = input['LI']
                HI = input['HI']

        self.real_HI = HI.to(self.device)
        self.real_LI = LI.to(self.device)

    def get_c(self,cdip,sdip):
        b, c, w, h = self.real_LI.shape
        return torch.zeros((b, cdip, w//sdip, h//sdip))

    def get_h(self,cdip,sdip):
        b, c, w, h = self.real_LI.shape
        return torch.ones((b, cdip, w//sdip, h//sdip))

    def init(self):
        self.t_h = {}#lstm last output
        self.t_c = {}#lstm state matrix
        self.t_h['1'] = self.get_h(64, 1)
        self.t_c['1'] = self.get_c(64, 1)

        self.t_h['2'] = self.get_h(128, 2)
        self.t_c['2'] = self.get_c(128, 2)

        self.t_h['3'] = self.get_h(128, 4)
        self.t_c['3'] = self.get_c(128, 4)

        self.t_h['4'] = self.get_h(256, 4)
        self.t_c['4'] = self.get_c(256, 4)

        self.boost_HI = self.real_LI.clone().detach()
        self.boost_HIS = [self.boost_HI]

    def forward(self):
        self.init()
        i = 0
        while i <= 2:
            self.boost_HI, self.t_h, self.t_c, self.boost_HI2, self.boost_HI4 = \
                                        self.netG(torch.cat((self.real_LI, self.boost_HIS[-1], self.boost_HIS[-1]), 1), self.t_h, self.t_c)
            self.boost_HIS.append(self.boost_HI)
            i += 1

        # test
        if not self.isTrain:
            for i in range(len(self.boost_HIS)):
                self.boost_HIS[i] = torch.clamp(self.boost_HIS[i], min=0, max=1)

            predict = tensor2im(self.boost_HIS[-1])
            target = tensor2im(self.real_HI)
            res = index.quality_assess(predict, target, self.loss_fn)
            print(res)
            self.avg_meters.update(res)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D_syn = self.backward_D_basic(self.netD, self.real_HI, self.boost_HI)

    def backward_G(self):

        self.loss_Pix = 0.0
        iter_num = len(self.boost_HIS)
        sigma = 0.8
        for i in range(iter_num):
            if i > 0:
                a = np.power(sigma, iter_num - 1 - i)
                self.loss_Pix += self.criterionmse(self.boost_HIS[i], self.real_HI) * a * 1.5

        self.loss_MP = self.criterionVgg(self.boost_HI, self.real_HI) + \
                 0.8 * self.criterionVgg(self.boost_HI2,self.real_HI2) + \
                 0.6 * self.criterionVgg(self.boost_HI4, self.real_HI4)

        self.loss_Adv = self.criterionGAN(self.netD(self.boost_HI), True) * 0.01

        self.loss_Grad = self.criterionGra(self.boost_HI, self.real_HI)

        self.loss_Color = self.criterionColor(self.boost_HI, self.real_HI) * 0.5


        self.loss_T = self.loss_Pix + self.loss_MP + self.loss_Adv + \
                        self.loss_Grad + self.loss_Color

        self.loss_T.backward()

    def optimize_parameters(self):

        self.optimizer_G.zero_grad()
        self.set_requires_grad([self.netD], False)
        self.forward()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


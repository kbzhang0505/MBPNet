import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .vgg import Vgg19
import torch.nn.functional as F

#Calculated gradient loss
class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.gradient = Gradient_Net(device)
    def compute_gradient(self, img):
        gradimg = self.gradient(img)
        return gradimg

    def forward(self, predict, target):
        predict_grad = self.compute_gradient(predict)
        target_grad = self.compute_gradient(target)

        return self.loss(predict_grad, target_grad)

class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = (0.3 * x[:, 0, :, :] + 0.59 * x[:, 1, :, :] + 0.11 * x[:, 2, :, :]).view(b, 1, h, w)
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient*0.5

#Calculate color loss
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, predict, target):
        b, c, h, w = target.shape
        target_view = target.view(b, c, h * w).permute(0, 2, 1)
        predict_view = predict.view(b, c, h * w).permute(0, 2, 1)
        target_norm = torch.nn.functional.normalize(target_view, dim=-1)
        predict_norm = torch.nn.functional.normalize(predict_view, dim=-1)
        cose_value = target_norm * predict_norm
        cose_value = torch.sum(cose_value, dim=-1)
        color_loss = torch.mean(1 - cose_value)

        return color_loss


class NoneNorm(torch.nn.Module):
    def __init__(self, *args):
        super(NoneNorm, self).__init__()
    def forward(self, x):
        return x

#Returns a standardization layer
def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = NoneNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer):

    def lambda_rule(epoch):
        if epoch > 50:
            lr_lambda = 0.5
        else:
            lr_lambda = 1
        return lr_lambda
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

#Initialize the network weight
def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):  # initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, init_type='normal', init_gain=0.02,gpu_ids=[]):

    net = Generator(input_nc, output_nc, ngf)

    return init_net(net, init_type, init_gain, gpu_ids)#Returns the initialized network


def define_D(input_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss1(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class LSTM64(nn.Module):
    def __init__(self, n_feats):
        super(LSTM64, self).__init__()

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats * 2 , n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):

        x = torch.cat((x, h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTM128(nn.Module):
    def __init__(self, n_feats):
        super(LSTM128, self).__init__()

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats*2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats * 4 , n_feats*2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats*2, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats*2, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):

        x = torch.cat((x, h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTM256(nn.Module):
    def __init__(self, n_feats):
        super(LSTM256, self).__init__()

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats * 8 , n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats*4, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):

        x = torch.cat((x, h), 1)

        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class Branch1(nn.Module):
    def __init__(self, n_feats):
        super(Branch1, self).__init__()

        self.se_layer = SELayer(n_feats, 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.lstm64 = LSTM64(n_feats)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):
        x = self.se_layer(x)

        x = self.conv1(x)
        x = self.conv2(x)

        h, c = self.lstm64(x, h, c)

        x = self.conv3(h)
        x = self.conv4(x)

        return x, h, c


class Branch2(nn.Module):
    def __init__(self, n_feats):
        super(Branch2, self).__init__()

        self.se_layer = SELayer(n_feats * 2, 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.lstm128_2 = LSTM128(n_feats)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):
        x = self.se_layer(x)

        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        res2 = x

        h, c = self.lstm128_2(x, h, c)

        h = h + res2
        x = self.conv3(h)
        x = x + res1
        x = self.conv4(x)

        return x, h, c


class Branch3(nn.Module):
    def __init__(self, n_feats):
        super(Branch3, self).__init__()

        self.se_layer = SELayer(n_feats * 2, 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.lstm128_4 = LSTM128(n_feats)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):

        x = self.se_layer(x)

        res1 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + res1 * 0.3

        h, c = self.lstm128_4(x, h, c)

        res2 = h
        x = self.conv3(h)
        x = self.conv4(x)
        x = x + res2 * 0.3

        return x, h, c


class Branch4(nn.Module):
    def __init__(self, n_feats):
        super(Branch4, self).__init__()

        self.se_layer = SELayer(n_feats * 4, 8)

        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.lstm256 = LSTM256(n_feats)

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):

        x = self.se_layer(x)

        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)

        x = self.conv7(x)

        h, c = self.lstm256(x, h, c)

        x = self.conv1(h)
        x = self.conv2(x)

        return x, h, c


class Generator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 1, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )

        ######
        self.branch1 = Branch1(n_feats)

        self.branch2 = Branch2(n_feats)

        self.branch3 = Branch3(n_feats)

        self.branch4 = Branch4(n_feats)
        ######

        self.branch2img = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )

        self.branch4img = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, h, c):
        x = self.conv1(x)#32,1
        x = self.conv2(x)#64,1
        res1 = x
        x = self.conv3(x)#128,1/2
        x = self.conv4(x)#128,1/2
        res2 = x
        x = self.conv5(x)#128,1/4
        res3 = x
        x = self.conv6(x)#256,1/4

        res1, h['1'], c['1'] = self.branch1(res1, h['1'], c['1'])#64,1

        res2, h['2'], c['2'] = self.branch2(res2, h['2'], c['2'])#128,1/2
        branch2img = self.branch2img(res2)

        res3, h['3'], c['3'] = self.branch3(res3, h['3'], c['3'])#128,1/4

        x, h['4'], c['4'] = self.branch4(x, h['4'], c['4'])#128,1/4

        x = x + res3
        branch4img = self.branch4img(x)

        x = self.conv9(x)#128,1/4

        x = self.deconv1(x)#128,1/2
        x = self.conv10(x)#128,1/2

        x = x + res2
        x = self.conv11(x)#128,1/2

        x = self.deconv2(x)#64,1
        x = self.conv12(x)#64,1

        x = x + res1
        x = self.conv13(x)#64,1
        x = self.conv14(x)#32,1

        x = self.output(x)#3,1

        return x, h, c, branch2img, branch4img
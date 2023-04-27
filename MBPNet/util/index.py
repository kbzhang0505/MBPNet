
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_psnr
# from skimage.measure import compare_ssim
from functools import partial
import numpy as np
import lpips

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))


def quality_assess(X, Y, loss_fn):

    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))

    img0 = lpips.im2tensor(X)
    img1 = lpips.im2tensor(Y)
    lpip = loss_fn.forward(img0, img1)

    return {'PSNR':psnr, 'SSIM': ssim, 'LPIPS': lpip.detach().item()}

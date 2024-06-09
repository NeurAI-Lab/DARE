import cv2
import torch
import numpy as np
from PIL import Image
from utils.loggers import plot
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

SAVE_SOBEL = False

class AuxiliaryNet():
    def __init__(self, args, dataset=None, device='cpu'):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.loss_type = self.args.loss_type
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.size = args.img_size
        self.transform = self.get_aux_transform()
        self.normalize = transforms.Normalize(mean=self.dataset.MEAN, std=self.dataset.STD)

    def get_aux_transform(self):

        if self.args.dataset == 'domain-net':
            transform = [transforms.ToPILImage(),
                 transform_sobel_edge(self.args, self.args.shape_upsample_size),
                 transforms.Resize((self.args.img_size, self.args.img_size)),
                 transforms.RandomCrop(self.args.img_size, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 ]
        elif self.args.dataset == 'rot-mnist' or self.args.dataset == 'perm-mnist':
            transform = [transforms.ToPILImage(),
                         transform_canny_edge(),
                         transforms.ToTensor()]
        else:
            transform = [transforms.ToPILImage(),
                 transform_sobel_edge(self.args, self.args.shape_upsample_size),
                 transforms.RandomCrop(self.size, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                ]

        if self.args.aug_norm:
            transform.append(self.normalize)

        return transforms.Compose(transform)

    def get_data(self, input):
        ret_tuple = torch.stack([self.transform(ee) for ee in input]).to(self.device)
        return ret_tuple


    def loss(self, out1, out2, feat1=None, feat2=None):
        if 'kl' in self.loss_type:
            loss = self.kl_loss(out1, out2)
        if 'l2' in self.loss_type:
            loss = self.l2_loss(out2, out1)
        return loss

    def kl_loss(self, out1, out2, T=1):
        p = F.log_softmax(out1 / T, dim=1)
        q = F.softmax(out2 / T, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * (T**2) / out1.shape[0]
        return l_kl

    def l2_loss(self, out1, out2):
        criterion_MSE = nn.MSELoss(reduction='mean')
        return criterion_MSE(out1, out2)

    def collate_loss(self, final_loss, loss_ce, loss_buf_ce=0, loss_aux=0, loss_aux_mem=0, loss_aux_buf=0, loss_logit_mem=0, m1=True):

        if m1:
            str = "m1"
        else:
            str = "m2"
        final_loss[str + '_loss_ce'] = loss_ce

        if loss_buf_ce is not None:
            final_loss[str + '_loss_buf_ce'] = loss_buf_ce
        if loss_aux is not None:
            final_loss[str + '_loss_aux'] = loss_aux
        if loss_aux_buf is not None:
            final_loss[str + '_loss_aux_buf'] = loss_aux_buf
        if loss_aux_mem is not None:
            final_loss[str + '_loss_aux_mem'] = loss_aux_mem
        if loss_logit_mem is not None:
            final_loss[str + '_loss_buf'] = loss_logit_mem

        return final_loss


class transform_canny_edge(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = Image.fromarray(cv2.bilateralFilter(np.array(img),5,75,75))
        gray_scale = transforms.Grayscale(1)
        image = gray_scale(img)
        edges = cv2.Canny(np.array(image), 100, 200)
        out = edges #np.stack([edges, edges, edges], axis=-1)
        to_pil = transforms.ToPILImage()
        out = to_pil(out)

        if SAVE_SOBEL:
            plot(out, 'canny')

        return out

class transform_sobel_edge(object):
    def __init__(self, args, upsample_size=0):
        self.gauss_ksize = args.sobel_gauss_ksize
        self.sobel_ksize = args.sobel_ksize
        self.upsample = args.sobel_upsample
        self.upsample_size = upsample_size

    def __call__(self, img, boxes=None, labels=None):

        if SAVE_SOBEL:
            plot(img, 'before_sobel')

        if self.upsample == 'True':
            curr_size = img.size[0]
            resize_up = transforms.Resize(max(curr_size, self.upsample_size), 3)
            resize_down = transforms.Resize(curr_size, 3)
            rgb = np.array(resize_up(img))
        else:
            rgb = np.array(img)

        if len(rgb.shape) != 3:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb = cv2.GaussianBlur(rgb, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)
        sobelx = cv2.Sobel(rgb, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        imgx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(rgb, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        imgy = cv2.convertScaleAbs(sobely)
        tot = np.sqrt(np.square(sobelx) + np.square(sobely))
        imgtot = cv2.convertScaleAbs(tot)
        sobel_img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))

        sobel_img = resize_down(sobel_img) if self.upsample == 'True' else sobel_img

        if SAVE_SOBEL:
            plot(sobel_img, 'sobel')

        return sobel_img

class transform_lowpass_fft(object):

    def __init__(self, args, size):
        self.args = args
        self.size = size
        #self.radius = args.radius

    def __call__(self, img):
        if SAVE_SOBEL:
            plot(img, 'before_fourier')

        r = 4 #self.radius  # how narrower the window is
        ham = np.hamming(self.size)[:, None]  # 1D hamming
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r  # expand to 2D hamming

        gray_image = np.array(img)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        f = cv2.dft(gray_image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
        f_filtered = ham2d * f_complex

        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        fourier_img = Image.fromarray(cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR))

        if SAVE_SOBEL:
            plot(fourier_img, 'fourier')

        return fourier_img

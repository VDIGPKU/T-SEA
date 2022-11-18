import kornia
import torch
import torch.nn.functional as F
import math
import numpy as np

from .convertor import FormatConverter
import kornia.augmentation as K


class DataTransformer(torch.nn.Module):
    """This achieves differentiable data augmentation based on Kornia (http://openaccess.thecvf.com/content_WACV_2020/papers/Riba_Kornia_an_Open_Source_Differentiable_Computer_Vision_Library_for_PyTorch_WACV_2020_paper.pdf).
        (ps:　Not used in T-SEA)"""
    def __init__(self, device: torch.device, rand_rotate: int = 10, rand_zoom_in: float = 0.3, rand_brightness: float = 0.2,
                 rand_saturation: float = 0.3,
                 rand_shift: float = 0.3):
        super().__init__()
        self.device = device
        self.rand_rotate_angle = rand_rotate
        self.rand_rotate = rand_rotate / 180 * math.pi
        self.rand_zoom_in = rand_zoom_in
        self.rand_brightness = rand_brightness
        self.rand_saturation = rand_saturation
        self.rand_shift = rand_shift

    def rand_affine_matrix(self, img_tensor):
        batch_size = img_tensor.size(0)
        # ----shift------
        tx = torch.cuda.FloatTensor(batch_size).fill_(0).uniform_(-self.rand_shift, self.rand_shift)
        ty = torch.cuda.FloatTensor(batch_size).fill_(0).uniform_(-self.rand_shift, self.rand_shift)
        # ----rotate-----
        angle = torch.cuda.FloatTensor(batch_size).uniform_(-self.rand_rotate, self.rand_rotate)
        # print('angle shape:', angle.shape)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # -------scale-------
        scale = torch.cuda.FloatTensor(batch_size).uniform_(1-self.rand_zoom_in, 1+self.rand_zoom_in)
        # scale = 1
        theta = torch.cuda.FloatTensor(batch_size, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, img_tensor.shape)
        img_tensor_t = F.grid_sample(img_tensor, grid)
        return img_tensor_t

    def forward(self, img_tensor: torch.tensor, p_aug: float = 0.5) -> torch.tensor:
        batch_size = img_tensor.size(0)
        gate = torch.zeros(1).bernoulli_(p_aug).item()
        if gate == 0:
            return img_tensor

        choice = int(torch.FloatTensor([0]).uniform_(0, 3.999))
        img_tensor_t = img_tensor
        if choice == 0:
            img_tensor_t = K.RandomGaussianNoise(mean=0., std=.01, p=.5)(img_tensor)
            factor = torch.cuda.FloatTensor(batch_size).fill_(0).uniform_(0, self.rand_brightness)
            img_tensor_t = kornia.enhance.adjust_brightness(img_tensor_t, factor, clip_output=True)
            # img_tensor_t = kornia.enhance.adjust_contrast(img_tensor_t, factor, clip_output=True)
        elif choice == 1:
            factor = torch.FloatTensor(batch_size).fill_(0).uniform_(1 - self.rand_saturation, 1 + self.rand_saturation)
            img_tensor_t = kornia.enhance.adjust_saturation(img_tensor, factor)
        elif choice == 2:
            # img_tensor_t = K.RandomRotation(self.rand_rotate_angle)(img_tensor)
            img_tensor_t = self.rand_affine_matrix(img_tensor)
        elif choice == 3:
            img_tensor_t = K.RandomGrayscale(p=1)(img_tensor)

        # torch.clamp_(img_tensor_t, min=0, max=1)
        return img_tensor_t


def mixup_transform(x1: torch.tensor,  cutmix_prob: int = 0.5, beta: int = 10, x2: torch.tensor = None) -> torch.tensor:
    """Mixup data augmentation(https://arxiv.org/pdf/1710.09412.pdf).
        (ps:　Not used in T-SEA)"""
    if np.random.rand() > cutmix_prob:
        return x1

    if x2 is None:
        batch_size = x1.size(0)
        indices = torch.randperm(batch_size, device=torch.device('cuda'))
        x2 = x1[indices, :, :, :].clone()
    # print(np.random.beta(beta, beta))
    lam = torch.cuda.FloatTensor(1).fill_(np.random.beta(beta, beta))

    x = lam * x1 + (1 - lam) * x2
    return x
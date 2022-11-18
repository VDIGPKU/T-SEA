import sys
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from natsort import natsorted

from .transformer import mixup_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DetDataset(Dataset):
    def __init__(self, images_path, input_size, is_augment=False, return_img_name=False):
        self.images_path = images_path
        self.imgs = os.listdir(images_path)
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        # is_augment = False
        self.transform = transforms.Compose([])
        if is_augment:
            self.transform = self.transform_fn
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.return_img_name = return_img_name

    def transform_fn(self, im, p_aug=0.5):
        """This is for random preprocesser augmentation of p_aug probability

        :param im:
        :param p_aug: probability to augment preprocesser.
        :return:
        """
        gate = torch.tensor([0]).bernoulli_(p_aug)
        if gate.item() == 0: return im
        im_t = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.RandomResizedCrop((416, 416), scale=(0.2, 0.9)),
            transforms.RandomRotation(5),
        ])(im)

        return im_t

    def pad_scale(self, img):
        """Padding the img to a square-shape to avoid stretch from the Resize op.

        :param img:
        :return:
        """
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad = (abs(pad_size), 0)
            side_len = h
        else:
            side_len = w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)
        return padded_img

    def __getitem__(self, index):
        # print(self.imgs[index], index)
        img_path = os.path.join(self.images_path, self.imgs[index])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = self.pad_scale(image)

        if self.return_img_name:
            return self.ToTensor(image), self.imgs[index]

        return self.ToTensor(image)

    def __len__(self):
        return self.n_samples


class DetDatasetLab(Dataset):
    """This is a Dataset with preprocesser label loaded."""

    def __init__(self, images_path, lab_path, input_size):
        self.im_path = images_path
        self.lab_path = lab_path
        self.labs = natsorted(filter(lambda p: p.endswith('.txt'), os.listdir(lab_path)))
        self.input_size = input_size
        self.max_n_labels = 10
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def pad_im(self, img, lab):
        """Padding the img to a square-shape and rescale the labels.

        :param img:
        :param lab:
        :return:
        """
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad_size = abs(pad_size)
            pad = (pad_size, 0)
            side_len = h
            lab[:, [1, 3]] = (lab[:, [1, 3]] * w + pad_size) / h
        else:
            side_len = w
            lab[:, [2, 4]] = (lab[:, [2, 4]] * h + pad_size) / w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)

        return padded_img, lab

    def batchify_lab(self, lab):
        """Padding to batchify the lab in length of (self.max_n_labels).

        :param lab:
        :return:
        """
        lab = torch.cat(
            (lab[:, 1:], torch.ones(len(lab)).unsqueeze(1), torch.zeros(len(lab)).unsqueeze(1)),
            1
        )
        # print('loader pad lab: ', lab)
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
            # padded_lab[-pad_size:, -1] = -1
        else:
            padded_lab = lab
        return padded_lab

    def __getitem__(self, index):
        lab_path = os.path.join(self.lab_path, self.labs[index])
        im_path = os.path.join(self.im_path, self.labs[index].replace('txt', 'png'))

        lab = np.loadtxt(lab_path) if os.path.getsize(lab_path) else np.zeros(5)
        lab = torch.from_numpy(lab).float()
        if lab.dim() == 1:
            lab = lab.unsqueeze(0)
        lab = lab[:self.max_n_labels]

        image = Image.open(im_path).convert('RGB')
        image, lab = self.pad_im(image, lab)

        return self.ToTensor(image), self.batchify_lab(lab)

    def __len__(self):
        return len(self.labs)


def check_valid(name: str):
    """To check if the file name is of a valid image format.

    :param name: file name
    :return: Boolean
    """
    return name.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))


def dataLoader(data_root, lab_root=None, input_size=None, batch_size=1, is_augment=False,
               shuffle=False, pin_memory=False, num_workers=16, sampler=None, return_img_name=False):
    if input_size is None:
        input_size = [416, 416]
    if lab_root is None:
        data_set = DetDataset(data_root, input_size, is_augment=is_augment, return_img_name=return_img_name)
    else:
        data_set = DetDatasetLab(data_root, lab_root, input_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    return data_loader


def read_img_np_batch(names, input_size):
    """Not used now.
    Read (RGB unit8) numpy img batch from names list and rescale into input_size
    This is now replaced by the DataLoader and Dataset for a faster I/O.

    :param names:
    :param input_size:
    :return: numpy, uint8, RGB, [0, 255], NCHW
    """
    img_numpy_batch = None
    for name in names:
        if not check_valid(name):
            print(f'{name} is invalid image format!')
            continue
        bgr_img_numpy = cv2.imread(name)
        # print('img: ', input_size)
        bgr_img_numpy = cv2.resize(bgr_img_numpy, input_size)
        img_numpy = cv2.cvtColor(bgr_img_numpy, cv2.COLOR_BGR2RGB)

        img_numpy = np.expand_dims(np.transpose(img_numpy, (2, 0, 1)), 0)

        # print(img_numpy.shape)
        if img_numpy_batch is None:
            img_numpy_batch = img_numpy
        else:
            img_numpy_batch = np.concatenate((img_numpy_batch, img_numpy), axis=0)
    return img_numpy_batch

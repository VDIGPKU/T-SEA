import numpy as np
import cv2
import torch


def random_transform(img_tensor, u=8):

    if np.random.random()>0.5:
        alpha=np.random.uniform(-u, u)/255
        img_tensor+=alpha
        img_tensor=img_tensor.clamp(min=-10, max=10)
    
    if np.random.random()>0.5:
        alpha=np.random.uniform(0.9, 1.1)
        img_tensor*=alpha
        img_tensor=img_tensor.clamp(min=-10, max=10)
    
    if np.random.random()>0.5:
        noise=torch.normal(0, 0.15, img_tensor.shape).cuda()
        img_tensor+=noise
        img_tensor=img_tensor.clamp(min=-10, max=10)

    return img_tensor


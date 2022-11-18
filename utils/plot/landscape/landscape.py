import os
from BaseDetectionAttack.tools.plot.landscape.D2Landscape import D2Landscape
import torch
import cv2
import numpy as np
import sys
sys.path.append('../../Draws/')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

total_image_in_set = 1000  # 训练集，测试集到底有多少image？


class Landscape():
    """
    这个类是封装画一张图的类，使得可以画多张图，既可以多张图画一起，也可以分开画
    """

    def __init__(self,
                 model,
                 patches_path: str,
                 output_path: str,
                 configs,
                 mode='3D',
                 iter_each_config=1,
                 one_image = False):
        """
        :param configs: for one figure, use multiple configs
        :param mode: 3D? Contour?
        :param one_image: only use one image to preprocesser?
        """
        self.patches_path = patches_path
        self.output_path = output_path
        self.args = configs
        self.mode = mode
        self.iter_each_config = iter_each_config
        self.one_image = one_image
        self.model = model

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def draw(self, total=10):
        """
        :param total:  how many patches you want to draw?
        """
        patches = os.listdir(self.patches_path)
        for i, patch in enumerate(patches):
            self._draw_one(patch, self.iter_each_config)
            if i > total:
                break

    def _draw_one(self, patch_name, step_each_point=1):
        patch = self.read_patch(self.patches_path + patch_name)
        use_image = np.random.randint(0, total_image_in_set) if self.one_image else None
        instance_train = D2Landscape(
            self.model(self.args[0], step_each_point, use_which_image=use_image),
            patch,
            mode=self.mode
        )
        coordinate = instance_train.synthesize_coordinates()
        if self.mode == '3D':
            figure = plt.figure()
            axes = Axes3D(figure)
            instance_train.draw(axes)
        else:
            instance_train.draw()
        for i, config in enumerate(self.args):
            if i >= 1:
                use_image = np.random.randint(0, total_image_in_set) if self.one_image else None
                instance_val = D2Landscape(
                    self.model(self.args[i], step_each_point, use_which_image=use_image),
                    patch,
                    mode=self.mode
                )
                instance_val.assign_coordinates(*coordinate)
                if self.mode == '3D':
                    instance_val.draw(axes)
                else:
                    instance_val.draw()
        plt.show()
        plt.savefig(self.output_path + patch_name + ".png")
        plt.clf()
        plt.close()

    @staticmethod
    def read_patch(patch_file):
        print('reading patch ' + patch_file)
        universal_patch = cv2.imread(patch_file)
        universal_patch = cv2.cvtColor(universal_patch, cv2.COLOR_BGR2RGB)
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).cuda()
        return universal_patch
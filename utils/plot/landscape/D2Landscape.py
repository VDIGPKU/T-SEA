import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from .ColorUtils import get_rand_cmap, suppress_stdout_stderr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

modes = ['3D', 'Contour', 'HeatMap', '2D']
alpha = 0.5  # 不透明度


class D2Landscape():
    """
    这个类负责画图，并且只负责画一张图
    """
    def __init__(self, model,
                 input: torch.tensor,
                 mode='3D'):
        """

        :param model: taken input as input, output loss
        :param input:
        """
        self.model = model
        self.input = input
        assert mode in modes
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def synthesize_coordinates(self,
                               x_min=-0.04, x_max=0.04, x_interval=0.001,
                               y_min=-0.04, y_max=0.04, y_interval=0.001):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.x, self.y = np.meshgrid(x, y)
        return self.x, self.y

    def assign_coordinates(self, x, y):
        self.x = x
        self.y = y

    def draw(self, axes=None):
        self._find_direction()
        z = self._compute_for_draw()
        self._draw3D(self.x, self.y, z, axes)

    def _find_direction(self):
        self.x0 = torch.randn(self.input.shape, device=self.device)
        self.y0 = torch.randn(self.input.shape, device=self.device)

    def _compute_for_draw(self):
        result = []
        if self.mode == '2D':
            self.x = self.x[0, :]
            for i in tqdm(range(self.x.shape[0])):
                with suppress_stdout_stderr():
                    now_x = self.x[i]
                    x = self.input + self.x0 * now_x
                    x = self.project(x)
                    loss = self.model(x)
                    result.append(loss)
        else:
            for i in tqdm(range(self.x.shape[0])):
                with suppress_stdout_stderr():
                    for j in range(self.x.shape[1]):
                        now_x = self.x[i, j]
                        now_y = self.y[i, j]
                        x = self.input + self.x0 * now_x + self.y0 * now_y
                        x = self.project(x)
                        loss = self.model(x)
                        result.append(loss)
        result = np.array(result)
        result = result.reshape(self.x.shape)
        return result

    def _draw3D(self, mesh_x, mesh_y, mesh_z, axes=None):
        """
        现在这个也能画2D了。。。
        """
        if self.mode == '3D':
            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')

        if self.mode == 'Contour':
            plt.contourf(mesh_x, mesh_y, mesh_z, 1, cmap=get_rand_cmap(), alpha=alpha)

        if self.mode == '2D':
            plt.plot(mesh_x, mesh_z)

        # plt.show()
        # plt.savefig(D2Landscape.get_datetime_str() + ".png")

    @staticmethod
    def get_datetime_str(style='dt'):
        import datetime
        cur_time = datetime.datetime.now()

        date_str = cur_time.strftime('%y_%m_%d_')
        time_str = cur_time.strftime('%H_%M_%S')

        if style == 'preprocesser':
            return date_str
        elif style == 'time':
            return time_str
        else:
            return date_str + time_str

    @staticmethod
    def project(x: torch.tensor, min=0, max=1):
        return torch.clamp(x, min, max)

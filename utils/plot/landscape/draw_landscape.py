import torch
from landscape import Landscape
import argparse


@torch.no_grad()
def train_valid_3dlandscape():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/test/test2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='3D',
                  iter_each_config=1)
    a.draw(total=10)

@torch.no_grad()
def train_valid_2dlandscape():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/test/test2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='2D',
                  iter_each_config=1)
    a.draw(total=10)

@torch.no_grad()
def multi_model_3dlandscape():
    '''
    通过传入多个config file来实现multi model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco3.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='3D',
                  iter_each_config=1)
    a.draw(total=10)

@torch.no_grad()
def multi_image_contourf(num_image=3):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_train = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train] * num_image,
                  mode='Contour',
                  iter_each_config=1,
                  one_image=True)
    a.draw(total=10)

@torch.no_grad()
def multi_model_contourf():
    '''
    通过传入多个config file来实现multi model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco3.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/preprocesser/coco/train/train2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='Contour',
                  iter_each_config=1)
    a.draw(total=10)

if __name__ == '__main__':
    train_valid_3dlandscape()
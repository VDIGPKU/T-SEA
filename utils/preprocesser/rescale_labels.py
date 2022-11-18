'''

split preprocesser for testing
'''
import os
import numpy as np
import random
import shutil
from tqdm import tqdm


def rescale_bbox_str(bbox: list, target_size: int, confidence: bool=False) -> list:
    """

    :param bbox: bbox array like ['cls name', x1, y1, x2, y2]
    :param target_size: target size to rescale (the rescale factor)
    :param confidence: if the bbox has confidence in it, if yes, then the bbox will be like ['cls name', conf, x1, y1, x2, y2]
    :return: rescaled bbox
    """
    x1_index = 1
    if confidence:
        x1_index = 2
    bbox[x1_index:] = [str(pos) for pos in np.array(bbox[x1_index:], dtype=float) * target_size]
    print(bbox)
    return bbox


def check(path, rebuild=False):
    if rebuild and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    datasets = ['coco', 'INRIAPerson']
    source_map = {'coco': ['train/train2017', 'val/val2017'], 'INRIAPerson': ['Test', 'Train']}

    targets = ['ground-truth', 'yolov3', 'yolov3-tiny', 'yolov4', 'yolov4-tiny', 'yolov5', 'faster_rcnn', 'ssd']
    targets = ['yolov3-tiny', 'yolov4', 'yolov4-tiny', 'yolov5', 'faster_rcnn', 'ssd']
    dataset = datasets[0]
    sources = source_map[dataset]
    sources = ['train/train2017']
    label_dir = 'labels'
    for source in sources:
        for target in targets:
            label_path = f'./{dataset}/{source}/{label_dir}/{target}-labels'
            save_dir = f'./{dataset}/{source}/{label_dir}/{target}-rescale-labels'
            check(save_dir)
            names = os.listdir(label_path)
            for name in tqdm(names):
                tmp = []
                with open(os.path.join(label_path, name), 'r') as f:
                    context = f.readlines()
                    # print(context)
                    for con in context:
                        bbox = con.replace('\n', '').split(' ')
                        rescaled_bbox = rescale_bbox_str(bbox, 416)
                        # print(rescaled_bbox)
                        tmp.append(' '.join(rescaled_bbox))
                res = '\n'.join(tmp)
                with open(os.path.join(save_dir, name), 'w') as f:
                    f.write(res)
                if len(tmp) > 5:
                    exit()
# check(save_dir, rebuild=True)
# check(save_label, rebuild=True)
#
# # def split_data(data_root, save_dir, file_num):
# img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
# if len(img_names) > file_num:
#     # print()
#     random.shuffle(img_names)
#     img_names = img_names[:file_num]
#
# for name in tqdm(img_names):
#     # save = name.replace('train', 'test')
#     cmd = 'cp ' + name + ' coco/test/test2017/'
#     # print(cmd)
#     os.system(cmd)
#     label = name.replace('train2017', 'labels').replace('.jpg', '.txt')
#     assert os.path.exists(label), f'Error, file {label} not exist!'
#
#     cmd = 'cp ' + label + ' coco/test/labels/'
#     os.system(cmd)
    # break
# print('written')
import os
import cv2
import numpy as np
import argparse
from natsort import natsorted

def p2v(path, fps=16, size=(400, 400), postfix='.png', save_path='./', save_name='video.avi'):
    filelist = natsorted(filter(lambda p: p.endswith(postfix), os.listdir(path)))
    video = cv2.VideoWriter(os.path.join(save_path, save_name),
                            cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    print(filelist)
    for file_name in filelist:
        item = os.path.join(path, file_name)
        img = cv2.imread(item)
        img = cv2.resize(img, size)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_name', default='video.avi')
    parser.add_argument('-p', '--path',
                        default='/home/chenziyan/work/BaseDetectionAttack/results/exp2/aug2/v3-300-300/patch',
                        type=str)
    parser.add_argument('-r', '--rule', type=str, default='.png')
    args = parser.parse_args()
    p2v(args.path, postfix=args.rule, save_name=args.save_name)
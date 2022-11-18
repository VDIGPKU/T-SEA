import argparse
import os
from tqdm import tqdm
import numpy as np
import re


def process_GT_label(args):
   source = args.source
   annotations_path = './INRIAPerson/'+source+'/annotations'
   annotations= os.listdir(annotations_path)
   save_path = './INRIAPerson/'+source+'/labels/ground-truth'
   os.makedirs(save_path, exist_ok=True)
   # print(annotations_path, save_path)
   # return
   str_XY = "(Xmax, Ymax)"
   str_size = 'Image size (X x Y x C)'

   os.makedirs(save_path, exist_ok=True)

   s = []
   for file in tqdm(annotations): #遍历文件夹
      with open(annotations_path+"/"+file, encoding = "ISO-8859-1") as f : #打开文件
         # print(annotations_path+"/"+file)
         clss = []
         bboxes = []
         iter_f = f.readlines()
         # print(iter_f)
         for line in iter_f:
            if str_size in line:
               size = [int(i) for i in re.findall(r"\d+", line)]
               # print('size: ', size)
               w, h = size[:2]
            if str_XY in line:
               # print(line)
               strlist = line.split(str_XY)
               strlist1 = "".join(strlist[1:])    # 把list转为str
               strlist1 = strlist1.replace(':', '')
               strlist1 = strlist1.replace('-', '')
               strlist1 = strlist1.replace('(', '')
               strlist1 = strlist1.replace(')', '')
               strlist1 = strlist1.replace(',', '')
               b = strlist1.split()
               bnd = np.array([float(b[0]) ,float(b[1]) ,float(b[2]) ,float(b[3])])
               bnd[[0, 2]] = bnd[[0, 2]] / w
               bnd[[1, 3]] = bnd[[1, 3]] / h
               bnd = [str(i) for i in bnd]
               # bnd.insert(0, 'person')
               bnd.insert(0, '0')
               # print('bnd: ', bnd)
               bboxes.append(' '.join(list(bnd)))
               # print('bbox: ', bboxes)
         with open(os.path.join(save_path, file), 'w') as f:
            f.write('\n'.join(bboxes))
      # break


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-s', '--source', type=str, default='Test')
   args = parser.parse_args()
   process_GT_label(args)

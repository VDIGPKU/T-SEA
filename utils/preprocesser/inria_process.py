import argparse
import os
from tqdm import tqdm
import numpy as np
import re
import shutil

def process_GT_label(args):
   annotations_path = args.annotations_path
   annotations= os.listdir(annotations_path)
   save_path = args.save_path
   if os.path.exists(save_path):
      shutil.rmtree(save_path)
   os.makedirs(save_path, exist_ok=True)
   # print(annotations_path, save_path)
   # return
   str_XY = "(Xmax, Ymax)"
   str_size = 'Image size (X x Y x C)'

   os.makedirs(save_path, exist_ok=True)

   for file in tqdm(annotations):
      with open(os.path.join(annotations_path, file), encoding = "ISO-8859-1") as f:
         bboxes = []
         iter_f = f.readlines()
         for line in iter_f:
            if str_size in line:
               size = [int(i) for i in re.findall(r"\d+", line)]
               w, h = size[:2]
            if str_XY in line:
               strlist = line.split(str_XY)
               strlist1 = "".join(strlist[1:])    # 把list转为str
               strlist1 = strlist1.replace(':', '')
               strlist1 = strlist1.replace('-', '')
               strlist1 = strlist1.replace('(', '')
               strlist1 = strlist1.replace(')', '')
               strlist1 = strlist1.replace(',', '')
               b = strlist1.split()
               bnd = np.array([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
               bnd[[0, 2]] = bnd[[0, 2]] / w
               bnd[[1, 3]] = bnd[[1, 3]] / h
               bnd *= args.rescale_factor
               bnd = [str(i) for i in bnd]
               bnd.insert(0, 'person')
               bboxes.append(' '.join(list(bnd)))
         with open(os.path.join(save_path, file), 'w') as f:
            f.write('\n'.join(bboxes))
      # break


if __name__ == "__main__":
   from pathlib import Path
   FILE = Path(__file__).resolve()
   PROJECT_DIR = FILE.parents[2]
   print(PROJECT_DIR)

   parser = argparse.ArgumentParser()
   parser.add_argument('-s', '--save_path', type=str, default=os.path.join(PROJECT_DIR, 'data/INRIAPerson/Test/labels/ground-truth-rescale-labels'),
                       help="Path of the parsed labels to save.")
   parser.add_argument('-a', '--annotations_path', type=str, default=os.path.join(PROJECT_DIR, 'data/INRIAPerson/Test/annotations'),
                       help="Path of the annotation files.")
   parser.add_argument('-r', '--rescale_factor', type=int, default=416,
                       help="Rescale factor for an input size [41, 416]. Decide this based on your input image size.")
   args = parser.parse_args()

   process_GT_label(args)

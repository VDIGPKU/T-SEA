import argparse
import cv2
import json
import shutil
import os
from tqdm import tqdm


class ConvertCOCOToYOLO:
    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """
    def __init__(self, img_folder, json_path, save_path, name_file):
        self.img_folder = img_folder
        self.json_path = json_path
        self.save_path = save_path
        self.names = load_class_names(name_file)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        self.sep = os.sep

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, f'Illegal path {img_path}, image is None!'
        return img.shape

    def convert_labels(self, img_path, x1, y1, x2, y2, resize=1):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
            else:
                lmax, lmin = l2, l1
            return lmax, lmin

        size = self.get_img_shape(img_path)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        xmin /= size[1] * resize
        xmax /= size[1] * resize
        ymin /= size[0] * resize
        ymax /= size[0] * resize
        return (xmin, ymin, xmax, ymax)

    def convert(self, annotation_key='annotations', img_id='image_id', cat_id='category_id', bbox_name='bbox', rescale_factor=1):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))
        
        check_set = set()

        # Retrieve preprocesser
        for key in tqdm(data[annotation_key]):
            # print(key)
            # Get required preprocesser
            image_id = f'{key[img_id]}'
            category_id = key[cat_id] - 1
            bbox = key[bbox_name]

            # Retrieve image.
            image_id = ('%12d' % int(image_id)).replace(' ', '0')
            image_path = f'{self.img_folder}{self.sep}{image_id}.jpg'


            # Convert the preprocesser: bbox [x, y, w, h] to bbox [x1, y1, x2, y2]
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])
            yolo_bbox *= rescale_factor

            # Prepare for export
            label_name = self.names[category_id].replace(' ', '')

            filename = f'{self.save_path}{self.sep}{image_id}.txt'
            content = f"{label_name} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"

            if label_name[0] == '-':
                content = ''
            # Export 
            if image_id in check_set:
                open_type = 'a'
            else:
                check_set.add(image_id)
                open_type = 'w'
            print(content)
            # with open(filename, open_type) as file:
            #     file.write(content)
            break
        self.check_empty_label()

    def check_empty_label(self):
        ims = os.listdir(self.img_folder)
        for im_name in tqdm(ims):
            label_name = im_name.split('.')[0] + '.txt'
            label_path = os.path.join(self.save_path, label_name)
            if not os.path.exists(label_path):
                f = open(label_path, 'w')
                f.close()
                print('Empty object: ', label_path)


# To run in as a class
if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
    sys.path.append(PROJECT_ROOT)
    print(PROJECT_ROOT, sys.path)
    from utils.parser import load_class_names

    # print(sys.path)
    target = 'val'
    postfix = '2017'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_folder', type=str, help='image dir', default=f'coco/{target}/{target}{postfix}')
    parser.add_argument('-n', '--name_file', type=str, help='class name file dir', default=f'configs/namefiles/coco80.names')
    parser.add_argument('-j', '--json_path', type=str, help='coco obj annotation .json file path',
                        default=f'./coco/instances_{target}{postfix}.json')
    parser.add_argument('-s', '--save_path', type=str, help='label save dir', default=f'./coco/{target}/{target}{postfix}-labels/ground-truth')
    parser.add_argument('-r', '--rescale_factor', type=int, default=416, help="Rescale factor for an input size [41, 416]. Decide this based on your input image size.")
    args = parser.parse_args()
    util = ConvertCOCOToYOLO(args.img_folder, args.json_path, args.save_path, args.name_file)
    util.convert(rescale_factor=args.rescale_factor)
    # util.check_empty_label()

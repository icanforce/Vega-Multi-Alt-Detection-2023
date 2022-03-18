import argparse
import glob
import os
from tqdm import tqdm
from PIL import Image
import re
import shutil
import json

def find_applicable_video_frames(image_fold, gt_fold, attr_fold, alt):

    attr_folders = glob.glob(os.path.join(attr_fold, "train", "*.txt"))
    attr_folders.extend(glob.glob(os.path.join(attr_fold, "test", "*.txt")))

    if alt == "high":
        alt_ind = 5
    elif alt == "medium":
        alt_ind = 4
    else:
        alt_ind = 3

    kept_video_folders = list()
    for img_path in attr_folders:
        with open(img_path, 'r') as file:
            line = file.readline().strip().split(',')
            if int(line[8]) == 1 and int(line[alt_ind]) == 1:
                kept_video_folders.append(os.path.basename(img_path).split("_")[0].strip())
            file.close()

    print("Kept folders: {}".format(kept_video_folders))

    image2annot = list()
    for video_folder in kept_video_folders:
        image_fps = glob.glob(os.path.join(image_fold, video_folder, "*.jpg"))
        annotations_fp = os.path.join(gt_fold, video_folder + "_gt_whole.txt")
        for image_fp in image_fps:
            image2annot.append((image_fp, annotations_fp))

    return image2annot

def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard_iou(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def create_bbox_text(image2annot):
    for index in tqdm(range(len(image2annot))):
        img_path, txt_path = image2annot[index]

        print(img_path, txt_path)

        img = Image.open(img_path).convert("RGB")

        # Bboxes and Label array must be parrallel.
        image_data = []

        regex = re.compile(r'\d+')
        matches = regex.finditer(os.path.basename(img_path))
        indices = next(matches).span()
        img_video_frame_id = int(os.path.basename(img_path)[indices[0] : indices[1]].lstrip('0'))

        with open(txt_path, 'r') as gt_file:
            gt_lines = gt_file.readlines()
            for line in gt_lines:
                line_split = line.strip().split(',')
                line_split = [int(v) for v in line_split]

                if int(line_split[0]) == img_video_frame_id:
                    '''Filter conditions'''
                    '''Category ids: (i.e.,car(1), truck(2), bus(3))'''
                    if int(line_split[-3]) == 1 and \
                    int(line_split[-2]) == 1 and \
                    int(line_split[-1]) in [1, 3]:
                        xmin, ymin = line_split[2], line_split[3]
                        xmax = xmin + line_split[4]
                        ymax = ymin + line_split[5]
                        img_bboxes_voc = [xmin, ymin, xmax, ymax]

                        if int(line_split[-1]) == 1:
                            '''This is encoding the car category as a 2'''
                            class_id = 2 # Changing class id from 1 -> 2 because the model has beeen tranined to map to 2.
                        else:
                            '''This is encoding the bus categpory as a 3'''
                            assert int(line_split[-1]) == 3
                            class_id = 3

                        img_bboxes_voc.append(class_id)
                        image_data.append(img_bboxes_voc)


        print(image_data)
        raise ValueError("h")






        # print(img_video_frame_id)
        # raise ValueError("H")

        # print(img_video_frame_id)

        # class_ids = list()
        #
        # with open(txt_path, 'r') as gt_file:
        #     gt_lines = gt_file.readlines()
        #     for line in gt_lines:
        #         line_split = line.strip().split(',')
        #         line_split = [int(v) for v in line_split]
        #
        #         if int(line_split[0]) == img_video_frame_id:
        #             '''Filter conditions'''
        #             '''Category ids: (i.e.,car(1), truck(2), bus(3))'''
        #             if int(line_split[-3]) == 1 and \
        #             int(line_split[-2]) == 1 and \
        #             int(line_split[-1]) in [1, 3]:

                        # print(line_split)
                        # ann_nest_dict = {}
                        # New_ann_id += 1
                        # x, y, w, h = int(line_split[2]), int(line_split[3]), int(line_split[4]), int(line_split[5])
                        # ann_nest_dict['bbox'] = [x, y, w, h]
                        # ann_nest_dict['id'] = New_ann_id
                        # ann_nest_dict['image_id'] = index
                        # ann_nest_dict['area'] = (x + w) * (y + h)
                        # ann_nest_dict['iscrowd'] = 0
                        # if int(line_split[-1]) == 1:
                        #     '''This is encoding the car category as a 2'''
                        #     ann_nest_dict['category_id'] = 2
                        # else:
                        #     '''This is encoding the bus categpory as a 3'''
                        #     assert int(line_split[-1]) == 3
                        #     ann_nest_dict['category_id'] = 3
                        # annotations_full_list.append(ann_nest_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "arguments to filter and mode UAVDT data")
    parser.add_argument('--image_fold', dest = 'image_fold', required = True)
    parser.add_argument('--attr_fold', dest = 'attr_fold', required = True)
    parser.add_argument('--gt_fold', dest = 'gt_fold', required = True)
    parser.add_argument('--alt', dest = 'alt', required = True, help = "Can either be high, medium, or low")

    args = parser.parse_args()

    image2annot = find_applicable_video_frames(args.image_fold, args.gt_fold, args.attr_fold, args.alt)
    create_bbox_text(image2annot)

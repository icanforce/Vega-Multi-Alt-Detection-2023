# Linear Regression (sklearn) y -> bbox size x -> confidence (poly, linear, SVM).
# Start off combining all low, high, medium

# Make on list for carpk and one list to UAVDT (Complete UAVDT)

# We ignored buses for now leading to no false positives.
# Also make sure to utilize defaults nms thresholding
'''
For personal_reference: python3 metric_categorization.py --image_fold /Users/vishal.jain/Documents/ICANFORCE_Drones/UAV-benchmark-M
--attr_fold /Users/vishal.jain/Documents/ICANFORCE_Drones/M_attr
--gt_fold /Users/vishal.jain/Documents/ICANFORCE_Drones/UAV-benchmark-MOTD_v1.0/GT
--alt medium --nms_thresh 0.01
'''

# Do Resizing and IOU
# Then look at how the resizing works

import argparse
import glob
import os
from tqdm import tqdm
from PIL import Image
import re
import shutil
import json
import cv2

import torch
import numpy as np

from effdet import create_model
from effdet.data import resolve_input_config
from timm.models.layers import set_layer_config
from contextlib import suppress
import albumentations

# For debugging and plotting
import matplotlib.pyplot as plt


def set_device(input_device):
    global device
    device = torch.device(input_device)
    print("Device: {}".format(input_device))

def create_effdet(args):

    args['pretrained'] = args['pretrained'] or not args['checkpoint']  # might as well try to validate something
    # create model
    with set_layer_config(scriptable=args['torchscript']):
        extra_args = {}
        if args['img_size'] is not None:
            extra_args = dict(image_size=(args['img_size'] ,args['img_size']))
        bench = create_model(
            args['model'],
            bench_task='predict',
            num_classes=args['num_classes'],
            pretrained=args['pretrained'],
            redundant_bias=args['redundant_bias'],
            soft_nms=args['soft_nms'],
            checkpoint_path=args['checkpoint'],
            checkpoint_ema=args['use_ema'],
            **extra_args,
        )

    model_config = bench.config
    input_config = resolve_input_config(args, model_config)

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args['model'], param_count))
    bench = bench.to(device)
    amp_autocast = suppress
    bench.eval()

    return bench, amp_autocast

# Image Preprocessing functions
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ImageToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size

class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        img_scale = 1. / img_scale  # back to original

        return new_img, img_scale

def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color

def transforms_coco_eval(
        img,
        img_size=512,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    transformed_img, scale = ResizePad(target_size=[img_size, img_size], interpolation=interpolation, fill_color=fill_color)(img)
    transformed_img = ImageToNumpy()(transformed_img)

    img_tensor = torch.zeros((1, *transformed_img.shape), dtype=torch.uint8)
    img_tensor[0] += torch.from_numpy(transformed_img)
    img_tensor = img_tensor.to(device)

    mean = torch.tensor([x * 255 for x in mean]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([x * 255 for x in std]).to(device).view(1, 3, 1, 1)

    img_tensor = img_tensor.float().sub_(mean).div_(std)

    return img_tensor, scale

def find_applicable_video_frames(image_fold, gt_fold, attr_fold, alt):

    attr_folders = glob.glob(os.path.join(attr_fold, "train", "*.txt"))
    attr_folders.extend(glob.glob(os.path.join(attr_fold, "test", "*.txt")))
    alt_ind_list = list()

    if alt == "high" or "high" in alt:
        alt_ind_list.append(5)
    if alt == "medium" or "medium" in alt:
        alt_ind_list.append(4)
    if alt == "low" or "low" in alt:
        alt_ind_list.append(3)

    kept_video_folders = list()
    for alt_ind in alt_ind_list:
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

def carpk_get_image2annot(data_fold):

    text_paths = glob.glob(os.path.join(data_fold, "Annotations", "*.txt"))
    image_paths = glob.glob(os.path.join(data_fold, "Images", "*png"))

    image_paths, text_paths = sorted(image_paths), sorted(text_paths)
    image2annot = list(zip(image_paths, text_paths))

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

def run_iou_thresh(out, iou_thresh):
    skipped_ind= list()
    for ii, pred in enumerate(out):
        iou = jaccard_iou(pred[:4].unsqueeze(0),out[:, :4])
        for jj, value in enumerate(iou[0][(ii + 1):]):
            if value > iou_thresh:
                skipped_ind.append(ii + jj + 1)

    skipped_ind, new_out, out = set(skipped_ind), list(), list(out)
    for i in range(len(out)):
        if i not in skipped_ind:
            new_out.append(out[i])

    new_out = torch.stack(new_out)
    return new_out

def def_args(checkpoint_file, model_name):

    args={}

    print("Pulling from model weights: {}".format(checkpoint_file))

    args['num_classes'] = 3
    args['pretrained'] = True
    args['checkpoint'] = checkpoint_file
    args['redundant_bias'] = None
    args['model'] = model_name
    args['soft_nms'] = None
    args['use_ema'] = True
    args['img_size'] = 896
    args['torchscript'] = True

    return args

def overlap_bbox(box1, box2):
    if (box1[0]>=box2[2]) or (box1[2]<=box2[0]) or (box1[3]<=box2[1]) or (box1[1]>=box2[3]):
        # return False if no overlapping
        return False
    #return true if overlapping
    return True


def gauge_performance(output, gt, total_bg, total_cfp, total_ctp, total_nfp, total_ntp):

    '''
    Classifies boxes as total_bg, total_cfp, total_ctp, total_nfp, total_ntp
    '''

    gt_tensor = torch.FloatTensor(gt)
    '''Create a dict that links every index to the categorization of the box'''
    ind2categ = dict()

    '''Iterate and classify every box in output'''
    for i, box in enumerate(output):
        box_class_id = int(box[-1])
        mask = list()
        for gt_box in gt_tensor:
            # iterating through every single ground truth bbox.
            mask.append(overlap_bbox(box, gt_box))
        if sum(mask) == 0:
            '''If the box does not overlap with any gt box then label box as bg and continue to next box'''
            total_bg += 1
            ind2categ[i] = "total_bg"
            continue
        '''Box has IOU with a gt box'''

        applicable_gt = gt_tensor[mask] # Index only boxes that overlap
        '''For each prediction box you have to find the closest GT boxes
        If closest has wrong class id then classify them as fp
        If closest has correct class id then classift them as tp
        If box overlaps with less than 0.05 thresh then is background.
        '''

        '''For all calculations here use max IOU -> torch.max(...) '''
        jaccard_scores = jaccard_iou(box[:-2].unsqueeze(0), applicable_gt[:, :-1])
        iou, pred_num = torch.max(jaccard_scores), torch.argmax(jaccard_scores)
        if iou.item() > 0.6:
            '''Can either be cfp or ctp'''
            if applicable_gt[pred_num][-1].item() == box_class_id:
                '''It is ctp because class ids match, then continue to next box'''
                total_ctp += 1
                ind2categ[i] = "total_ctp" # Use this
                continue
            else:
                '''It is cfp because class ids do not match, then continue to next box'''
                total_cfp += 1
                ind2categ[i] = "total_cfp"
                continue
        elif iou.item() > 0.05 and iou.item() < 0.6:
            '''Can either be nfp or ntp'''
            if applicable_gt[pred_num][-1].item() == box_class_id:
                '''It is ntp because class ids match, then continue to next box'''
                total_ntp += 1
                ind2categ[i] = "total_ntp" # Use this
                continue
            else:
                '''It is nfp because class ids do not match, then continue to next box'''
                total_nfp += 1
                ind2categ[i] = "total_nfp"
                continue
        else:
            total_bg += 1
            ind2categ[i] = "total_bg" # Use this
            continue

    return total_bg, total_cfp, total_ctp, total_nfp, total_ntp, ind2categ

def vis_complex_stats(image, ind2categ, output):
    '''Should only be applied to the output of model'''
    '''Order of color categ KEY: total_nfp, total_ntp, total_cfp, total_ctp, total_bg'''
    '''Very IMPORTANT: BGR ColorWay'''
    categ_COLORS = {"total_nfp": (0, 0, 0), "total_ntp": (0, 255, 0), "total_cfp": (0, 0, 255), "total_ctp": (225, 225, 225), "total_bg": (255, 0, 0)} #BGR colorway
    for i, box in enumerate(output):
        color = categ_COLORS[ind2categ[i]]
        if ind2categ[i] in ["total_ntp", "total_ctp", "total_bg", "total_nfp", "total_cfp"]:
            print(ind2categ[i], color)
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
    return image

def draw_boxes(boxes, labels, image, COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    # read the image with OpenCV
    for i, box in enumerate(boxes):
        color = COLORS[labels[i] % len(COLORS)]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
    return image

def per_image_stats(ind2categ, output, image_fp):
    img_basename = os.path.basename(image_fp)
    with open("per_image_stats.txt", 'a+') as f:
        for ind in list(ind2categ.keys()):
            box = output[ind][:5]
            box_categ = ind2categ[ind]
            class_id = output[ind][-1]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if class_id == 2:
                desc_name = "car"
            else:
                continue # ignore buses

            final_string = "{},{},{},{},{},{},{},{},{}".format(
                img_basename,
                box_categ,
                box[0].item(),
                box[1].item(),
                box[2].item(),
                box[3].item(),
                box[4].item(),
                area,
                desc_name,
                )

            print(final_string, file=f)

def create_UAVDT(img_path, txt_path):

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
                '''Filter conditions. This is where you can have occlusion params. No filtering for now'''
                '''Category ids: (i.e.,car(1), truck(2), bus(3))'''
                if int(line_split[-1]) in [1, 3]:
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

    return image_data

def create_CarpK(img_path, txt_path):

    image_data = []

    with open(txt_path, 'r') as gt_file:
        gt_lines = gt_file.readlines()
        for line in gt_lines:
            line_split = line.strip().split()
            xmin, ymin, xmax, ymax = int(line_split[0]), int(line_split[1]), int(line_split[2]), int(line_split[3])
            # Encode a two in the end because class id is always 2
            img_bboxes_voc = [xmin, ymin, xmax, ymax, 2]
            image_data.append(img_bboxes_voc)

    return image_data

def process_Carpk_img(img, image_data):
    transform = albumentations.Compose(
                [albumentations.augmentations.transforms.Resize(img.height, img.width)],
                bbox_params=albumentations.BboxParams(format='pascal_voc'))
    blank_sheet = np.zeros((img.height, img.width))
    for i, bbox in enumerate(image_data):
        color_pixel = i + 1
        if color_pixel >= 255:
            raise ValueError("Too many cars")
        blank_sheet[bbox[1]: bbox[3], bbox[0] : bbox[2]] = int(color_pixel)
    horizontal_pad, vertical_pad = int(img.width), int(img.height)
    np_image = np.asarray(img)
    image_data = np.asarray(image_data)
    # Visualize bboxes next
    np_image = np.pad(np_image,
                        ((vertical_pad, vertical_pad),  # pad bottom
                         (horizontal_pad, horizontal_pad),  # pad right
                         (0, 0)),  # don't pad channels
                        mode='constant',
                        constant_values=0)

    padded_blank_sheet = np.pad(blank_sheet,
                        ((vertical_pad, vertical_pad),
                          (horizontal_pad, horizontal_pad)),
                        mode='constant',
                        constant_values=0)
    unique_masks = np.unique(padded_blank_sheet)
    scaled_bboxes = list()
    for i, bbox_id in enumerate(unique_masks[1:]):
        single_mask = np.where(padded_blank_sheet == bbox_id, padded_blank_sheet, padded_blank_sheet * 0)
        bbox_indices = np.where(single_mask == bbox_id)

        y_min, y_max = bbox_indices[0][0], bbox_indices[0][-1]
        x_min, x_max = bbox_indices[1][0], bbox_indices[1][-1]

        bbox = [x_min, y_min, x_max, y_max, image_data[i]]
        if x_max > x_min and y_max > y_min:
            scaled_bboxes.append(bbox)

    transformed = transform(image = np_image, bboxes = scaled_bboxes)
    final_image, final_bboxes = transformed["image"], transformed["bboxes"]
    final_image = Image.fromarray(np.uint8(final_image))
    boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3]), 2] for box in final_bboxes]
    # final_bboxes = []
    # for box in boxes:
    #     try:
    #         p_image = cv2.rectangle(final_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 4)
    #         final_bboxes.append(box)
    #     except:
    #         continue
    # p_image = cv2.rectangle(final_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 4)
    return final_image, boxes


def create_bbox_text(image2annot, args, nms_thresh, iou_thresh):
    total_bg = 0 # Total background detections (No IOU)
    total_nfp = 0 # Total near false positives (little IOU, false classID)
    total_ntp = 0 # Total near true positive (little IOU, true classID)
    total_cfp = 0 # Total clear false positives (big IOU, false classID)
    total_ctp = 0 # Total clear true positives (big IOU, correct classID)

    if os.path.exists("per_image_stats.txt"):
        print("Text file already exists will prepeare to delete file")
        os.remove("per_image_stats.txt")
    file = open('per_image_stats.txt', 'w')
    file.write("img_bn,categ,xmin,ymin,xmax,ymax,conf,area,classID\n")
    file.close()

    for index in tqdm(range(len(image2annot))):
        img_path, txt_path = image2annot[index]

        print(img_path, txt_path)

        img = Image.open(img_path).convert("RGB")
        # Bboxes and Label array must be parrallel.
        if "CARPK" in img_path:
            image_data = create_CarpK(img_path, txt_path)
            '''Create code to process the image sizes'''
            img, image_data = process_Carpk_img(img, image_data) # pascal_voc
        elif "UAV-benchmark" in txt_path:
            image_data = create_UAVDT(img_path, txt_path)
        else:
            raise RuntimeError("Can't classify dataset")
        # print(img.width, img.height)

        transformed_frame, img_scale = transforms_coco_eval(img, args["img_size"])
        output = bench(transformed_frame)[0]
        final_out = list()
        for ii, pred in enumerate(output):
            #Nonmax Suppression
            if pred[-2] > nms_thresh:
                final_out.append(pred)
            else:
                break

        if len(final_out) != 0:
            final_out = torch.stack(final_out)
        if len(final_out) > 1:
             #Nonmax Suppression
             final_out = run_iou_thresh(final_out, iou_thresh)
        else:
            final_out = []

        if len(final_out) != 0:
            final_out[:, :-2] = final_out[:, :-2] * img_scale
            final_out = final_out[~(final_out[:, -1] == 1.0)]



        total_bg, total_cfp, total_ctp, total_nfp, total_ntp, ind2categ = gauge_performance(final_out,
                                                           image_data,
                                                           total_bg,
                                                           total_cfp,
                                                           total_ctp,
                                                           total_nfp,
                                                           total_ntp)
        per_image_stats(ind2categ, final_out, img_path)


        # if index == 0:
        #     '''Only do visualization on first iteration of image'''
        #     print("Visualize an Example Ground Truth")
        #     image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        #
        #     tensor_img_data = torch.tensor(image_data)
        #     # image = draw_boxes(final_out[:, :-2], final_out[:, -1].to(int), image)
        #     cv2.imshow('gt image',draw_boxes(tensor_img_data[:, :-1], tensor_img_data[:, -1], image.copy()))
        #     cv2.waitKey()
        #
        #     print("Visualize Predictions and their Bounding Box Categories")
        #     # Deal with empty outputs
        #     cv2.imshow('out image', vis_complex_stats(image, ind2categ, final_out[:, :-1]))
        #     cv2.waitKey()
        #
        #     print(total_nfp, total_ntp, total_cfp, total_ctp, total_bg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "arguments to filter and mode UAVDT data")
    parser.add_argument('--image_fold', dest = 'image_fold', required = True)
    parser.add_argument('--attr_fold', dest = 'attr_fold', required = True)
    parser.add_argument('--gt_fold', dest = 'gt_fold', required = True, help = "Must be in form of UAV-benchmark-MOTD_v1.0/GT")
    parser.add_argument('--alt', dest = 'alt', required = True, help = "Can either be high, medium, or low")
    # parser.add_argument('--weight', dest = 'weight', required = True, help = "Weights for your model")
    # parser.add_argument('--model_name, dest = model_name', required = True, help = "name of effecientdet model")
    parser.add_argument('--device', dest = 'device', required = False, help = "marks the device")
    parser.add_argument('--nms_thresh', dest = "nms_thresh", required = True, help = "refers to the nms thresh")
    parser.add_argument('--iou', dest = "iou_thresh", required = False, type = float, default = 0.7, help = "iou thresh for post-processing bboxes")
    parser.add_argument('--model_kind', dest = "model_kind", required = False, default = "d3", help = "model to choose for bbox stats")
    parser.add_argument('--carpk', dest = "carpk", required = False, help = "root path of carpk dataset (must have ending of CARPK_devkit/data)")
    # parser.add_argument('--input', dest = input_file, required = False, help = "Text file in format [image_basename, xmin, ymin, xmax, ymax, class_id]")
    args = parser.parse_args()

    if args.model_kind == "d3":
        weight, model_name = "finetuned_d3_model_best.pth.tar", "tf_efficientdet_d3"
    elif args.model_kind == "d0":
        weight, model_name = "d0_march.pth.tar", "efficientdet_d0"


    if args.device:
        set_device(args.device)
    else:
        set_device("cpu")
        print("Defaulting to CPU")

    image2annot = list() # A list of all img path to label pairs
    # UAVDT is off
    # UAVDT_image2annot = find_applicable_video_frames(args.image_fold, args.gt_fold, args.attr_fold, args.alt)
    # image2annot.extend(UAVDT_image2annot)

    if args.carpk:
        carpk_image2annot = carpk_get_image2annot(args.carpk)
        image2annot.extend(carpk_image2annot)

    import random
    random.shuffle(image2annot)

    model_args = def_args(weight, model_name)
    bench, amp_autocast = create_effdet(model_args)
    create_bbox_text(image2annot, model_args, float(args.nms_thresh), args.iou_thresh)

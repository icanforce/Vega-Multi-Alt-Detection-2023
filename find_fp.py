# Create functionaolity to lower nms if bbox is empty
#For every false positive, can you print the confidence value?
# Text function

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
    Counts the amount of bg boxes from an out and gt
    Counts the amount of cfp boxes from an out and gt
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
                ind2categ[i] = "total_ctp"
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
                ind2categ[i] = "total_ntp"
                continue
            else:
                '''It is nfp because class ids do not match, then continue to next box'''
                total_nfp += 1
                ind2categ[i] = "total_nfp"
                continue
        else:
            total_bg += 1
            ind2categ[i] = "total_bg"
            continue

    return total_bg, total_cfp, total_ctp, total_nfp, total_ntp, ind2categ

def vis_complex_stats(image, ind2categ, output):
    '''Should only be applied to the output of model'''
    '''Order of color categ KEY: total_nfp, total_ntp, total_cfp, total_ctp, total_bg'''
    categ_COLORS = {"total_nfp": (255, 0, 0), "total_ntp": (0, 255, 0), "total_cfp": (0, 0, 255), "total_ctp": (225, 225, 225), "total_bg": (0, 0, 0)}
    for i, box in enumerate(output):
        color = categ_COLORS[ind2categ[i]]
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

def create_bbox_text(image2annot, args, nms_thresh, iou_thresh):
    total_bg = 0 # Total background detections (No IOU)
    total_nfp = 0 # Total near false positives (little IOU, false classID)
    total_ntp = 0 # Total near true positive (little IOU, true classID)
    total_cfp = 0 # Total clear false positives (big IOU, false classID)
    total_ctp = 0 # Total clear true positives (big IOU, correct classID)

    for index in tqdm(range(len(image2annot))):
        img_path, txt_path = image2annot[index]

        print(img_path, txt_path)

        img = Image.open(img_path).convert("RGB")

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

        total_bg, total_cfp, total_ctp, total_nfp, total_ntp, ind2categ = gauge_performance(final_out,
                                                           image_data,
                                                           total_bg,
                                                           total_cfp,
                                                           total_ctp,
                                                           total_nfp,
                                                           total_ntp)

        if index == 0:
            '''Only do visualization on first iteration of image'''
            print("Visualize an Example Ground Truth")
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            tensor_img_data = torch.tensor(image_data)
            # image = draw_boxes(final_out[:, :-2], final_out[:, -1].to(int), image)
            cv2.imshow('gt image',draw_boxes(tensor_img_data[:, :-1], tensor_img_data[:, -1], image.copy()))
            cv2.waitKey()

            print("Visualize Predictions and their Bounding Box Categories")
            cv2.imshow('out image', vis_complex_stats(image, ind2categ, final_out[:, :-1]))
            cv2.waitKey()

            print(total_nfp, total_ntp, total_cfp, total_ctp, total_bg)



        raise ValueError("h")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "arguments to filter and mode UAVDT data")
    parser.add_argument('--image_fold', dest = 'image_fold', required = True)
    parser.add_argument('--attr_fold', dest = 'attr_fold', required = True)
    parser.add_argument('--gt_fold', dest = 'gt_fold', required = True)
    parser.add_argument('--alt', dest = 'alt', required = True, help = "Can either be high, medium, or low")
    # parser.add_argument('--weight', dest = 'weight', required = True, help = "Weights for your model")
    # parser.add_argument('--model_name, dest = model_name', required = True, help = "name of effecientdet model")
    parser.add_argument('--device', dest = 'device', required = False, help = "marks the device")
    parser.add_argument('--nms_thresh', dest = "nms_thresh", required = True, help = "refers to the nms thresh")
    parser.add_argument('--iou', dest = "iou_thresh", required = False, type = float, default = 0.7, help = "iou thresh for post-processing bboxes")
    # parser.add_argument('--input', dest = input_file, required = False, help = "Text file in format [image_basename, xmin, ymin, xmax, ymax, class_id]")
    weight, model_name = "finetuned_d3_model_best.pth.tar", "tf_efficientdet_d3"


    args = parser.parse_args()

    if args.device:
        set_device(args.device)
    else:
        set_device("cpu")
        print("Defaulting to CPU")

    image2annot = find_applicable_video_frames(args.image_fold, args.gt_fold, args.attr_fold, args.alt)
    model_args = def_args(weight, model_name)
    bench, amp_autocast = create_effdet(model_args)
    create_bbox_text(image2annot, model_args, float(args.nms_thresh), args.iou_thresh)

import argparse
import glob
import os
from tqdm import tqdm
from PIL import Image
import re
import shutil
import json

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


def count_tp(output, gt):

    # Maybe it has close iou with many boxes. To get highest IOU
    gt_tensor = torch.FloatTensor(gt)
    for box in output:
        box_class_id = int(box[-1])
        #knock out options in gt, so you limit IOU calculations, so we are creating a condensed gt
        cond_gt = gt_tensor[gt_tensor[:, -1] == box_class_id]
        mask = list()
        for gt_box in cond_gt:
            mask.append(overlap_bbox(box, gt_box))
        applicable_gt = cond_gt[mask]

        print(box)
        print(applicable_gt)

        # Check to see if IOU calculated from IOU func match with overlap results


        raise ValueError("jdhdwh")


    pass



def create_bbox_text(image2annot, args, nms_thresh, iou_thresh):
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

        print(final_out)

        # Bboxes and Label array must be parrallel.
        image_data = []

        regex = re.compile(r'\d+')
        matches = regex.finditer(os.path.basename(img_path))
        indices = next(matches).span()
        img_video_frame_id = int(os.path.basename(img_path)[indices[0] : indices[1]].lstrip('0'))

        print(img_video_frame_id)

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


        print(image_data)
        # Visualize pred and gt data through cv2
        count_tp(final_out, image_data)
        # Check which boxes are correct
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
    # parser.add_argument('--weight', dest = 'weight', required = True, help = "Weights for your model")
    # parser.add_argument('--model_name, dest = model_name', required = True, help = "name of effecientdet model")
    parser.add_argument('--device', dest = 'device', required = False, help = "marks the device")
    parser.add_argument('--nms_thresh', dest = "nms_thresh", required = True, help = "refers to the nms thresh")
    parser.add_argument('--iou', dest = "iou_thresh", required = False, type = float, default = 0.7, help = "iou thresh for post-processing bboxes")
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

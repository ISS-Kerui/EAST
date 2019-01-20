# import sys
# sys.path.append('...')

from YOLO.YOLO_models import Darknet
from YOLO.YOLO_utils.utils import load_classes, non_max_suppression, src_nms
from YOLO.YOLO_utils.datasets import ImageFolder
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
import math
from YOLO import YOLO_config



def YOLO_preload():
    device = torch.device(YOLO_config.GPU_ID if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(YOLO_config.config_path, img_size=YOLO_config.img_size).to(device)
    # load yolo weights of torch version
    model.load_state_dict(torch.load(YOLO_config.weights_path))
    model.eval()
    print("model loaded.")
    return model

def YOLO_detect_one(model, image_folder1, image_folder2, image_folder3):

    device = torch.device(YOLO_config.GPU_ID if torch.cuda.is_available() else "cpu")

    # dataloader
    dataloader_lst = []
    dataloader1 = DataLoader(ImageFolder(image_folder1, img_size=YOLO_config.img_size), batch_size=YOLO_config.batch_size,
                             shuffle=False, num_workers=YOLO_config.n_cpu)
    dataloader_lst.append((dataloader1, YOLO_config.slide_size1, YOLO_config.stride_szie1))

    dataloader2 = DataLoader(ImageFolder(image_folder2, img_size=YOLO_config.img_size), batch_size=YOLO_config.batch_size,
                             shuffle=False, num_workers=YOLO_config.n_cpu)
    dataloader_lst.append((dataloader2, YOLO_config.slide_size2, YOLO_config.stride_szie2))

    dataloader3 = DataLoader(ImageFolder(image_folder3, img_size=YOLO_config.img_size), batch_size=YOLO_config.batch_size,
                             shuffle=False, num_workers=YOLO_config.n_cpu)
    dataloader_lst.append((dataloader3, YOLO_config.slide_size3, YOLO_config.stride_szie3))

    classes = load_classes(YOLO_config.class_path)

    Tensor = torch.FloatTensor

    print('\nPerforming object detection:')

    src_detections = []
    for (dataloader_i, slide_size_i, stride_size_i) in dataloader_lst:
        imgs = []
        img_detections = []
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader_i):

            input_imgs = Variable(input_imgs.type(Tensor))
            input_imgs = input_imgs.to(device)

            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 2, YOLO_config.conf_thres, YOLO_config.nms_thres)

                imgs.extend(img_paths)
                img_detections.extend(detections)

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            if detections is not None:
                box_i = eval(path.split('_')[-2])
                box_j = eval(path.split('_')[-1].split('.')[0])
                detections[:, 0] = detections[:, 0] * slide_size_i / YOLO_config.img_size
                detections[:, 1] = detections[:, 1] * slide_size_i / YOLO_config.img_size
                detections[:, 2] = detections[:, 2] * slide_size_i / YOLO_config.img_size
                detections[:, 3] = detections[:, 3] * slide_size_i / YOLO_config.img_size

                detections[:, 0] = detections[:, 0] + box_i * stride_size_i
                detections[:, 2] = detections[:, 2] + box_i * stride_size_i
                detections[:, 1] = detections[:, 1] + box_j * stride_size_i
                detections[:, 3] = detections[:, 3] + box_j * stride_size_i
                src_detections.append(detections)
    pre = []
    if len(src_detections) != 0:
        src_detections2 = torch.cat([src_detections[i] for i in range(len(src_detections))], 0)
        src_detections2 = src_detections2.unsqueeze(0)
        src_detections_ = src_nms(src_detections2, YOLO_config.nms_thres)


        if src_detections_[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in src_detections_[0]:
                # print('\t+ Label: %s, Conf: %.5f, cls_conf: %.5f' % (
                # classes[int(cls_pred)], conf.item(), cls_conf.item()))
                if cls_pred.item() == 0.0:
                    print('\t+ Label: %s, Conf: %.5f, cls_conf: %.5f' % (
                        classes[int(cls_pred)], conf.item(), cls_conf.item()))
                    pre.append((x1, y1, x2, y2))
    return pre

def process(pre, image_path, save_path):
    new_pos = []
    img = cv2.imread(image_path)
    if len(pre) >= YOLO_config.pre_thres:
        for (x1, y1, x2, y2) in pre:
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            item = 0
            h, w, _ = img.shape
            thres = math.sqrt(h ** 2 + w ** 2) * 0.2

            for i in pre:
                x_ = (i[0] + i[2]) / 2
                y_ = (i[1] + i[3]) / 2

                s = math.sqrt((x - x_) ** 2 + (y - y_) ** 2)
                if s < thres:
                    item = item + 1
            if item >= YOLO_config.item_thres:
                new_pos.append((x1, y1, x2, y2))

    if len(new_pos) != 0:
        for (x1, y1, x2, y2) in new_pos:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imwrite(save_path, img)
    return image_path, new_pos





if __name__ == '__main__':
    slide_image_root1 = "/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide1"
    slide_image_root2 = "/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide2"
    slide_image_root3 = "/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide3"
    image_root = "/home/hxj/Desktop/12.21/satellite_data/beijingdaxing"
    save_root = "/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_output8"

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    # load model
    model = YOLO_preload()

    # for img_dir in os.listdir(slide_image_root1):
    #     slide_image_path1 = os.path.join(slide_image_root1, img_dir)
    #     slide_image_path2 = os.path.join(slide_image_root2, img_dir)
    #     slide_image_path3 = os.path.join(slide_image_root3, img_dir)
    #
    #     image_path = os.path.join(image_root, img_dir + '.png')
    #     save_path = os.path.join(save_root, img_dir + '.png')
    #
    #     YOLO_pre = YOLO_detect_one(model, slide_image_path1, slide_image_path2, slide_image_path3)
    #     process(YOLO_pre, image_path, save_path)

    pre = YOLO_detect_one(model, '/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide1/39.65795153682715_116.2600577790175_18',
               '/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide2/39.65795153682715_116.2600577790175_18',
               '/home/hxj/Desktop/12.21/satellite_data/beijingdaxing_slide3/39.65795153682715_116.2600577790175_18')
    process(pre, '/home/hxj/Desktop/12.21/satellite_data/beijingdaxing/39.65795153682715_116.2600577790175_18.png',
               '/home/hxj/Desktop/12.21/satellite_data/6.png')

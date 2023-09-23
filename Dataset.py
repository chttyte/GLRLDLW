import os
import copy
import random

import PIL
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

def L_loader(path):
    return Image.open(path).convert('L')

def RGB_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, imgs, labels, bboxs, landmarks, flag, transform=None, target_transform=None, loader=RGB_loader): # flag是训练集或测试集的标签，loader是图片的加载模式
        self.imgs = imgs            # list
        self.labels = labels        # list
        self.bboxs = bboxs          # list
        self.landmarks = landmarks  # list
        self.transform = transform
        self.target_transform = target_transform #todo: 什么时候要用到？
        self.loader = loader
        self.flag = flag
        self.transform_strong = transforms.Compose([
            # transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
            transforms.Resize((112, 112)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def __getitem__(self, index):
        img_index = index
        img, label, bbox, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), \
                                     copy.deepcopy(self.bboxs[index]), copy.deepcopy(self.landmarks[index])
        ori_img_w, ori_img_h = img.size # 获取图片的长，高
        # BoundingBox，获取对角点的坐标数值
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True # ! 咩用

        if self.flag == 'train': # 训练集的预处理方式
            random_crop = True
            random_flip = True
        elif self.flag == 'test':
            random_crop = False
            random_flip = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(0.5 * max(0, int(0.20 * (bottom - top))))
    
        if enlarge_bbox: # 扩大bbox
            left = max(left - padding_w, 0) # 减，但是不能减到小于0
            right = min(right + padding_w, ori_img_w) # 加，但是不能加到超过原本图片的宽度

            top = max(top - padding_h, 0)   # 同理
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop: # 自定义随机裁框的大小
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img.crop((left, top, right, bottom))  # 裁剪得到新的图片
        crop_img_w, crop_img_h = img.size           # 裁剪后的高宽

        landmark[:, 0] -= left
        landmark[:, 1] -= top  # 求出五个标注点对应于裁剪后的图的位置

        if random_flip and random.random() > 0.5: # 将图片随机翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:, 0] = (right - left) - landmark[:, 0] # 翻转之后，五个标注点也要进行翻转

        # Transform Image
        trans_img = self.transform(img)
        # img_aug
        img_aug = self.transform_strong(img)
        _, trans_img_w, trans_img_h = trans_img.size()

        inputSizeOfCropNet = 28 #todo: 这个是怎么得出来的
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w # 放大？
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(np.int)

        grid_len = 7 #todo: 代表什么意思
        half_grid_len = int(grid_len/2) #todo: 有什么几何意义吗？

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1


        return img_index, trans_img, landmark, label, img_aug
        # 返回的是（图片的序号，一张裁剪过后并且进行精细定位和随机翻转等预处理里后的人脸图，该图的五个关键点，该图所属的表情标签）

    def __len__(self): 
        return len(self.imgs)


class MyDataset2(data.Dataset):
    def __init__(self, imgs, labels, bboxs, landmarks, true_label, flag, transform=None, target_transform=None,
                 loader=RGB_loader):  # flag是训练集或测试集的标签，loader是图片的加载模式
        self.imgs = imgs  # list
        self.labels = labels  # list
        self.bboxs = bboxs  # list
        self.landmarks = landmarks  # list
        self.transform = transform
        self.target_transform = target_transform  # todo: 什么时候要用到？
        self.loader = loader
        self.flag = flag
        self.true_label = true_label
        # self.transform_strong = transforms.Compose([
        #     transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        self.transform_strong = copy.deepcopy(transform)
        self.transform_strong.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        img_index = index
        img, label, bbox, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), \
                                     copy.deepcopy(self.bboxs[index]), copy.deepcopy(self.landmarks[index])
        true_labels = copy.deepcopy(self.true_label[index])
        ori_img_w, ori_img_h = img.size  # 获取图片的长，高
        # BoundingBox，获取对角点的坐标数值
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True  # ! 咩用

        if self.flag == 'train':  # 训练集的预处理方式
            random_crop = True
            random_flip = True
        elif self.flag == 'test':
            random_crop = False
            random_flip = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(
            0.5 * max(0, int(0.20 * (bottom - top))))

        if enlarge_bbox:  # 扩大bbox
            left = max(left - padding_w, 0)  # 减，但是不能减到小于0
            right = min(right + padding_w, ori_img_w)  # 加，但是不能加到超过原本图片的宽度

            top = max(top - padding_h, 0)  # 同理
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:  # 自定义随机裁框的大小
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img.crop((left, top, right, bottom))  # 裁剪得到新的图片
        crop_img_w, crop_img_h = img.size  # 裁剪后的高宽

        landmark[:, 0] -= left
        landmark[:, 1] -= top  # 求出五个标注点对应于裁剪后的图的位置

        if random_flip and random.random() > 0.5:  # 将图片随机翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:, 0] = (right - left) - landmark[:, 0]  # 翻转之后，五个标注点也要进行翻转

        # Transform Image
        trans_img = self.transform(img)
        # img_aug
        img_aug = self.transform_strong(img)
        _, trans_img_w, trans_img_h = trans_img.size()

        inputSizeOfCropNet = 28  # todo: 这个是怎么得出来的
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w  # 放大？
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(np.int)

        grid_len = 7  # todo: 代表什么意思
        half_grid_len = int(grid_len / 2)  # todo: 有什么几何意义吗？

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1

        return img_index, trans_img, landmark, label, true_labels, img_aug
        # 返回的是（图片的序号，一张裁剪过后并且进行精细定位和随机翻转等预处理里后的人脸图，该图的五个关键点，该图所属的表情标签）

    def __len__(self):
        return len(self.imgs)

##########################
def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v

    return img.rotate(v)





def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img

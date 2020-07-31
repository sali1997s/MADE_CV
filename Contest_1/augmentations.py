from random import random

import cv2
import torch
import numpy as np



class ScaleMinSideToSize(object):
    def __init__(self, size=(128, 128), elem_name='image'):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample['scale_coef'] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class HorisontalFlip(object):
    """Осуществляет поворот изображения вокруг горизонтальной оси"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random() < self.p:
            sample['image'] = sample['image'][:, ::-1, :]

            if 'landmarks' in sample:
                landmarks = sample['landmarks'].reshape(-1, 2)
                landmarks[:, 0] = sample['image'].shape[1] - landmarks[:, 0]
                sample['landmarks'] = landmarks.reshape(-1)

        return sample


class RandomRotation(object):
    """Осуществляет поворот изображения относительно центра со случайным углом и scaling'ом картинки"""

    def __init__(self, p=0.5, angle=(-10, 10), scale=(0.9, 1.1)):
        self.p = p
        self.angle = angle
        self.scale = scale

    def __call__(self, sample):
        if random() < self.p:
            image, landmarks = sample['image'], sample.get('landmarks', None)

            # Генерируем матрицу поворота
            angle, scale = np.random.uniform(*self.angle), np.random.uniform(*self.scale)
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, scale)
            M_torch = torch.tensor(M, dtype=torch.float32)

            # Поворачиваем изображение
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            sample['image'] = image

            # Поворачиваем ключевые точки
            if 'landmarks' in sample:
                landmarks = landmarks.reshape(-1, 2).float()
                landmarks = torch.cat((landmarks, torch.ones(size=(landmarks.shape[0], 1))), dim=1)
                landmarks = torch.mm(landmarks, M_torch.T).reshape(-1).to(torch.int16)
                sample['landmarks'] = landmarks

        return sample


class Grayscale(object):
    """Осуществляет перевод картинки в оттенки серого"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random() < self.p:
            sample['image'] = cv2.cvtColor(cv2.cvtColor(sample['image'], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return sample


class RandomConvertScaleAbs(object):
    """Осуществляет случайное изменение яркости и контраста"""
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

    def __init__(self, p=0.5, alpha=(0.9, 1.1), beta=(-10, 10)):
        self.p = p
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        if random() < self.p:
            alpha, beta = np.random.uniform(*self.alpha), np.random.uniform(*self.beta)
            print(alpha, beta)
            sample['image'] = cv2.convertScaleAbs(sample['image'], alpha=alpha, beta=beta)
        return sample


class RandomGammaCorrection(object):
    """Осуществляет случайную гамма-коррекцию"""
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

    def __init__(self, p=0.5, gamma=(0.5, 1.5)):
        self.p = p
        self.gamma = gamma

    def __call__(self, sample):
        if random() < self.p:
            gamma = np.random.uniform(*self.gamma)
            lookup_table = np.clip(np.power(np.arange(256).reshape(1, -1) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
            sample['image'] = cv2.LUT(sample['image'], lookup_table)
        return sample

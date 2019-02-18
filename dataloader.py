from os.path import isdir, exists, abspath, join

import numpy as np
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, root_dir="data", batch_size=16, iteration=100, input_size=(128, 128)):
        self.batch_size = batch_size
        self.iteration = iteration
        self.input_size = input_size
        self.root_dir = abspath(root_dir)

    def __iter__(self):
        current = 0
        endId = 0
        if self.mode == 'train':
            endId = self.iteration
        elif self.mode == 'test':
            endId = 1

        while current < endId:
            data_image_array = []
            gt_array = []
            for i_itr in range(self.batch_size):
                if self.mode == 'train':
                    img = Image.open(join(self.root_dir, "train.png"))
                else:
                    img = Image.open(join(self.root_dir, "test.png"))

                if self.mode == 'train':
                    img= self.__applyDataAugmentation(img)
                else:
                    # todo: add data augmentation for test case
                    img= self.__applyDataAugmentation(img)

                gt_img = np.array(img, dtype=np.float32) / 255
                gt_array.append(gt_img)

                data_image = np.array(img, dtype=np.float32)
                mask = np.ones(self.input_size)
                for i in range(5):
                    mask, data_image = self.__generateMask(mask, data_image)

                # origin_img_data = np.array(origin_img, dtype=np.float32)
                # plt.subplot(1, 3, 1)
                # plt.imshow(gt_img)
                # plt.subplot(1, 3, 2)
                # plt.imshow(data_image/255)
                # plt.subplot(1, 3, 3)
                # plt.imshow(mask)
                # plt.show()

                data_image /= 255

                # stack RGB data and mask data
                mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

                data_image = np.concatenate((data_image, mask), axis=2)
                data_image_array.append(data_image)

            current += 1
            data_image_np_array = np.array(data_image_array)
            gt_np_array = np.array(gt_array)
            yield (data_image_np_array, gt_np_array)

    def setMode(self, mode):
        self.mode = mode

    def __applyDataAugmentation(self, img):
        if random.random() > 0.5:
            img = self.__horizontalFlip(img)
        if random.random() > 0.5:
            img = self.__verticalFlip(img)
        if random.random() > 0.5:
            img = self.__colorJitter(img)
        if random.random() > 0.5:
            img = self.__rotate(img)
        img = self.__randomResizing(img)
        img = self.__randomCrop(img)
        return img

    def __horizontalFlip(self, img):
        horizontalFlip = transforms.RandomHorizontalFlip(p=1.0)
        return horizontalFlip(img)

    def __verticalFlip(self, img):
        verticalFlip = transforms.RandomVerticalFlip(p=1.0)
        return verticalFlip(img)

    def __colorJitter(self, img):
        # todo: decide parameter
        colorJitter = transforms.ColorJitter()  # parameter undecided
        return colorJitter(img)

    def __rotate(self, img):
        # todo: decide random angle
        angle = 15 * random.random()
        return transforms.functional.rotate(img, angle)

    def __randomResizing(self, img):
        w, h = img.size
        newW = 0
        newH = 0
        # make sure the resized img is larger than input_size for random crop
        while newW < self.input_size[1] or newH < self.input_size[0]:
            coefficent = 2 * random.random()
            newW = int(coefficent * w)
            newH = int(coefficent * h)
        return img.resize((newW, newH))

    def __randomCrop(self, img):
        randomCrop = transforms.RandomCrop(self.input_size)
        return randomCrop(img)

    def __generateMask(self, mask, img):
        if random.random() > 0.5:
            # 8x64 rectangle
            mask_w = 8
            mask_h = 64
        else:
            mask_w = 64
            mask_h = 8
        start_w = random.randrange(mask.shape[1] - mask_w)
        start_h = random.randrange(mask.shape[0] - mask_h)
        for i in range(start_w, start_w+mask_w):
            for j in range(start_h, start_h+mask_h):
                mask[j, i] = 0
                for k in range(3):
                    img[j, i, k] = 0
        return mask, img

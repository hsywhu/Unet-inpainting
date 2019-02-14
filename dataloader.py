import os
from os.path import isdir, exists, abspath, join

import numpy as np
import random
from PIL import Image
from torchvision import transforms
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.2):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            img = Image.open(self.data_files[current])
            img = img.resize((388, 388))
            origin_img = img

            label = Image.open(self.label_files[current])
            label = label.resize((388, 388))

            if self.mode == 'train':
                img, label = self.__applyDataAugmentation(img, label)

            data_image = np.array(img, dtype=np.float32)
            label_image = np.array(label, dtype=np.float32)

            if self.mode == 'train':
                if random.random() > 0.5:
                    data_image, label_image = self.__elasticDeformation(data_image, label_image)

            # seamless tiling input image
            data_image_ = np.zeros((572, 572), dtype=np.float32)
            data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2): int(data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) / 2), int((data_image_.shape[1] - data_image.shape[1]) / 2): int(data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) / 2)] = data_image
            for i in range(int((data_image_.shape[0] - data_image.shape[0]) / 2)):
                for j in range(int((data_image_.shape[1] - data_image.shape[1]) / 2)):
                    data_image_[i, j] = data_image_[(data_image_.shape[0] - data_image.shape[0]) - i - 1, (data_image_.shape[1] - data_image.shape[1]) - j - 1]
                    data_image_[data_image_.shape[0] - 1 - i, j] = data_image_[int(data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + i), (data_image_.shape[1] - data_image.shape[1]) - j - 1]
                    data_image_[i, data_image_.shape[1] - 1 - j] = data_image_[(data_image_.shape[0] - data_image.shape[0]) - i - 1, int(data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) + j)]
                    data_image_[data_image_.shape[0] - 1 - i, data_image_.shape[1] - 1 - j] = data_image_[data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + i, data_image_.shape[1] - (data_image_.shape[1] - data_image.shape[1]) + j]
            for i in range(data_image.shape[0]):
                for j in range(int((data_image_.shape[0] - data_image.shape[0]) / 2)):
                    data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), j] = data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - data_image.shape[0] - j - 1]
                    data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - j - 1] = data_image_[int((data_image_.shape[0] - data_image.shape[0]) / 2 + i), data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + j]
                    data_image_[j, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)] = data_image_[data_image_.shape[0] - data_image.shape[0] - j - 1, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)]
                    data_image_[data_image_.shape[0] - j - 1, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)] = data_image_[data_image_.shape[0] - (data_image_.shape[0] - data_image.shape[0]) + j, int((data_image_.shape[0] - data_image.shape[0]) / 2 + i)]

            # temp_image = Image.fromarray(data_image_)
            # temp_image.show()

            # origin_img_data = np.array(origin_img, dtype=np.float32)
            # plt.subplot(1, 2, 1)
            # plt.imshow(origin_img_data)
            # plt.subplot(1, 2, 2)
            # temp_img, temp_label = self.__elasticDeformation(origin_img_data, label_image)
            # plt.imshow(temp_img)
            # plt.show()

            data_image_ /= 255
            current += 1
            if self.mode == 'test':
                yield (data_image_, label_image, data_image)
            else:
                yield (data_image_, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def __applyDataAugmentation(self, img, label):
        if random.random() > 0.5:
            img, label = self.__horizontalFlip(img, label)
        if random.random() > 0.5:
            img, label = self.__zoom(img, label)
        if random.random() > 0.5:
            img, label = self.__rotate(img, label)
        if random.random() > 0.5:
            img, label = self.__gammaCorrectin(img, label)
        return img, label

    def __horizontalFlip(self, img, label):
        horizontalFlip = transforms.RandomHorizontalFlip(p=1.0)
        return horizontalFlip(img), horizontalFlip(label)

    def __zoom(self, img, label):
        w, h = img.size
        img = img.resize((int(h*1.05), int(w*1.05)))
        label = label.resize((int(h * 1.05), int(w * 1.05)))
        new_w, new_h = img.size
        start_y = random.randrange(new_h - h)
        start_x = random.randrange(new_w - w)
        return transforms.functional.crop(img, start_y, start_x, h, w), transforms.functional.crop(label, start_y, start_x, h, w)

    def __rotate(self, img, label):
        angle = 15 * random.random()
        return transforms.functional.rotate(img, angle), transforms.functional.rotate(label, angle)

    def __gammaCorrectin(self, img, label):
        gamma = random.randint(200, 500) / 100
        return transforms.functional.adjust_gamma(img, gamma, gain = 1), label

    def __elasticDeformation(self, img, label):
        return self.__elastic_transform(img, 128, 15), self.__elastic_transform(label, 128, 15)

    def __elastic_transform(self, image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        assert len(image.shape) == 2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape)

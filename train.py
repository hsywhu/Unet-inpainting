import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

# settings on mac
# from Unet.model import UNet
# from Unet.dataloader import DataLoader

# settings on windows/ubuntu
from model import UNet
# from unet import UNet
from dataloader import DataLoader

def train_net(net,
              epochs=5,
              data_dir="data",
              iteration=100,
              gpu=True):
    loader = DataLoader(data_dir, iteration=iteration)

 
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.99,
    #                       weight_decay=0.0005)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, img in enumerate(loader):
            # output image size: (N, H, W, C)
            img_tensor = torch.from_numpy(img).float()
            # turn image tensor from (N, H, W, C) to (N, C, H, W)
            img_tensor = img_tensor.permute(0, 3, 1, 2)

            if gpu:
                img_tensor = img_tensor.cuda()

            pred = net(img_tensor)
            img_gt = img_tensor[:, :3, :, :]
            loss = criterion(pred, img_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
 
            print('Training sample %d / %d - Loss: %.6f' % (i+1, iteration, loss.item()))

            # optimize weights

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

    # displays test images with original and predicted masks after training
    # loader.setMode('test')
    # net.eval()
    # with torch.no_grad():
    #     for _, (img, label, origin_img) in enumerate(loader):
    #         shape = img.shape
    #         img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
    #         if gpu:
    #             img_torch = img_torch.cuda()
    #         pred = net(img_torch)
    #         pred_sm = softmax(pred)
    #         _,pred_label = torch.max(pred_sm,1)
    #
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(origin_img)
    #         plt.subplot(1, 3, 2)
    #         plt.imshow((label-1)*255.)
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
    #         plt.show()
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        gpu=args.gpu,
        data_dir=args.data_dir)

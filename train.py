from os.path import join
from optparse import OptionParser
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from model import UNet
from resnet import resNet
from dataloader import DataLoader

def train_net(net,
              epochs=100,
              data_dir="data",
              iteration=100,
              gpu=True):
    loader = DataLoader(data_dir, iteration=iteration)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, gt) in enumerate(loader):
            # output image size: (N, H, W, C)
            img_tensor = torch.from_numpy(img).float()
            gt_tensor = torch.from_numpy(gt).float()
            # turn image tensor from (N, H, W, C) to (N, C, H, W)
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            gt_tensor = gt_tensor.permute(0, 3, 1, 2)
            if gpu:
                img_tensor = img_tensor.cuda()
                gt_tensor = gt_tensor.cuda()

            pred = net(img_tensor)
            loss = criterion(pred, gt_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 20 == 0:
                print('Training sample %d / %d - Loss: %.6f' % (i+1, iteration, loss.item()))

        # save train_groundtruth, train_input, train_output, test_groundtruth, test_input, test_output to samples folder
        for save_i, (img, gt) in enumerate(loader):
            # output image size: (N, H, W, C)
            im = Image.fromarray(np.uint8(gt[0, :, :, :].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_train_gt.png")
            im = Image.fromarray(np.uint8(img[0, :, :, :3].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_train_in.png")

            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            if gpu:
                img_tensor = img_tensor.cuda()
            pred = net(img_tensor)
            pred = pred.permute(0, 2, 3, 1)
            im = Image.fromarray(np.uint8(pred.cpu().detach().numpy()[0, :, :, :].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_train_out.png")
            if save_i == 0:
                break
        net.eval()
        loader.setMode('test')
        for _, (img, gt) in enumerate(loader):
            # output image size: (N, H, W, C)
            im = Image.fromarray(np.uint8(gt[0, :, :, :].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_test_gt.png")
            im = Image.fromarray(np.uint8(img[0, :, :, :3].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_test_in.png")

            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            if gpu:
                img_tensor = img_tensor.cuda()
            pred = net(img_tensor)
            pred = pred.permute(0, 2, 3, 1)
            im = Image.fromarray(np.uint8(pred.cpu().detach().numpy()[0, :, :, :].squeeze() * 255))
            im.save("sample/" + str(epoch + 1) + "_test_out.png")

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
            print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-n', '--network', dest='network', default='resNet', help='choose from UNet and resNet')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    if args.network == 'UNet':
        net = UNet()
    elif args.network == 'resNet':
        net = resNet()

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

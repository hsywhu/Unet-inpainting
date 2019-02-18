from dataloader import DataLoader
from model import UNet
import matplotlib.pyplot as plt
import torch
print("rendering")
# displays test images with original and predicted masks after training
data_dir='data/'
# model_dir='data/cells/checkpoints/CP58.pth'
# model_dir='data/aug/CP19_best_with_elastic.pth'
model_dir='data/checkpoints/CP12.pth'

gpu = True

net = UNet()
if gpu:
    net.load_state_dict(torch.load(model_dir))
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))

if gpu:
    net.cuda()

loader = DataLoader(data_dir)
loader.setMode('test')
net.eval()
with torch.no_grad():
    for _, (img, gt) in enumerate(loader):
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        if gpu:
            img_tensor = img_tensor.cuda()
        pred = net(img_tensor)

        img_tensor = img_tensor.permute(0, 2, 3, 1)
        plt.subplot(1, 3, 1)
        img_gt = gt[0, :, :, :].squeeze()
        plt.imshow(img_gt)
        plt.subplot(1, 3, 2)
        train_in = img_tensor.cpu().detach().numpy()[0, :, :, 0:3].squeeze()
        plt.imshow(train_in)
        plt.subplot(1, 3, 3)
        pred = pred.permute(0, 2, 3, 1)
        pred_np = pred.cpu().detach().numpy()[0, :, :, :].squeeze()
        plt.imshow(pred_np)
        plt.show()

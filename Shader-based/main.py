import pickle

import torch
from PIL import Image
import numpy
from matplotlib import pyplot as plt
import os
import cv2
import CnnModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', DEVICE)


def get_xy_torch(color_path, depth_path):
    color_image_pil = Image.open(color_path)
    depth_image_pil = Image.open(depth_path)
    color_image_numpy = numpy.array(color_image_pil)
    depth_image_numpy = numpy.array(depth_image_pil)
    color_image_numpy = cv2.cvtColor(color_image_numpy, cv2.COLOR_RGBA2RGB)
    depth_image_numpy = cv2.cvtColor(depth_image_numpy, cv2.COLOR_RGBA2GRAY)
    # color_image_numpy = cv2.resize(color_image_numpy, (640, 360))
    # depth_image_numpy = cv2.resize(depth_image_numpy, (640, 360))
    color_image_numpy = numpy.swapaxes(color_image_numpy, 2, 1)
    color_image_numpy = numpy.swapaxes(color_image_numpy, 1, 0)

    x_torch = torch.from_numpy(color_image_numpy).float()
    y_torch = torch.from_numpy(depth_image_numpy).float()
    return x_torch, y_torch


# color_dir = 'data/train/1/color'
# depth_dir = 'data/train/1/depth'
# xs_torch, ys_torch = [], []
# for file_name in os.listdir(color_dir):
#     if file_name.endswith('.png'):
#         x_torch, y_torch = get_xy_torch(os.path.join(color_dir, file_name), os.path.join(depth_dir, file_name))
#         xs_torch.append(x_torch)
#         ys_torch.append(y_torch)
#         print(file_name, 'added')
#
# os.makedirs('work', exist_ok=True)
# with open('work/xy.pkl', 'wb') as f:
#     pickle.dump(xs_torch, f)
#     pickle.dump(ys_torch, f)

with open('work/xy2.pkl', 'rb') as f:
    xs_torch = pickle.load(f)
    ys_torch = pickle.load(f)

h, w = ys_torch[0].shape

model = CnnModel.CnnModel(w * h).to(DEVICE)


def train():
    loss_fn = torch.nn.MSELoss().to(DEVICE)
    optim = torch.optim.Adam(model.parameters())

    model.train()
    sum_loss = 0
    i = 0
    for e in range(10000):
        index = numpy.random.randint(0, len(xs_torch))
        x_torch = xs_torch[index].unsqueeze(0).to(DEVICE)
        y_torch = ys_torch[index].unsqueeze(0).to(DEVICE)

        optim.zero_grad()
        y_hat_torch = model(x_torch)
        # print(y_torch.shape, y_hat_torch.shape)
        loss = loss_fn(y_torch, y_hat_torch)
        loss.backward()
        optim.step()

        sum_loss += loss.item()
        i += 1
        if e % 100 == 0:
            print(e, round(sum_loss / i, 3))
            sum_loss = 0
            i = 0
            # y_hat_torch = y_hat_torch[0][0].detach().cpu()
            # y_hat_numpy = numpy.array(y_hat_torch)
            # plt.imshow(y_hat_numpy)
            # plt.show()
            os.makedirs('work/32_2/', exist_ok=True)
            torch.save(model.state_dict(), 'work/32/'+str(e)+'.pkl')
            print(str(e), 'saved to', 'work/32/'+str(e)+'.pkl')


#train()


def eval():
    os.makedirs('output/1_2', exist_ok=True)
    model.load_state_dict(torch.load('work/32/9900.pkl'))
    model.eval()
    for i in range(len(xs_torch)):
        y_hat_torch = model(xs_torch[i].unsqueeze(0).cuda())
        y_hat_numpy = numpy.array(y_hat_torch.squeeze().detach().cpu())
        # y_hat_numpy = cv2.cvtColor(y_hat_numpy, cv2.COLOR_GRAY2RGBA)
        y_hat_numpy = (y_hat_numpy * 255).astype(int)
        output = Image.fromarray(y_hat_numpy)
        output.save('output/1_2/'+str(i)+'.png')

eval()


img = numpy.array(Image.open('data/train/1/vanilla_depth/0.png'))
print(numpy.max(img), numpy.min(img))
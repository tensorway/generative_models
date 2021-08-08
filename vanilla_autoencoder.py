#%%
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)

train_loader = th.utils.data.DataLoader(dataset1, batch_size=64, shuffle=False)
test_loader = th.utils.data.DataLoader(dataset2)

#%%
class AutoencoderMLP(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, x):
        h = x.view(x.shape[0], -1)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        h = th.sigmoid(h)
        return h.view(x.shape)

model = AutoencoderMLP([28*28, 64, 16, 2, 16, 64, 28*28])

opt = th.optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.NLLLoss()
# %%
writer = SummaryWriter()
nepoch = 3
for ep in range(nepoch):
    for i, (img, _) in enumerate(train_loader):
        pred_img = model(img)
        # loss = loss_func(pred_img, img)
        loss = ((pred_img-img)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('loss', loss, i+ep*len(train_loader))
        if i % 100 == 0:
            print(ep, i, loss)



# %%
import random
for img, _ in test_loader:
    if random.random() > 0.05:
        continue
    pimg = model(img)
    toplot = th.cat((img[0, 0], pimg[0, 0]), dim=1).detach().numpy()
    break
import matplotlib.pyplot as plt

plt.imshow(toplot)
# %%
pimg[0, 0]
img[0, 0]
img.shape
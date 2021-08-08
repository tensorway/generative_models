#%%
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from models import ConvNet, ConvTransposeNet, MLP
import matplotlib.pyplot as plt


transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('../data', train=False,               transform=transform)

train_loader = th.utils.data.DataLoader(dataset1, batch_size=16, shuffle=False)
test_loader = th.utils.data.DataLoader(dataset2)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
device
#%%
class ConvClassifier(nn.Module):
    def __init__(self, conv_arch, mlp_arch, last_activation=lambda x: F.softmax(x, dim=1)):
        super().__init__()
        self.conv_net = ConvNet(conv_arch, last_activation=F.relu)
        self.mlp      = MLP(mlp_arch, last_activation=last_activation)
    def forward(self, x, debug=False):
        h = self.conv_net(x, debug)
        h = h.view(h.shape[0], -1)
        out = self.mlp(h)
        return out

class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, post_latent_shape, net_arch, last_activation=lambda x: x, crop_func=lambda x:x, noise_device=th.device('cpu')):
        super().__init__()
        post_latent_dim = 1
        for item in post_latent_shape:
            post_latent_dim *= item
        self.post_latent_dim = post_latent_dim
        self.post_latent_shape = post_latent_shape
        self.project_latent = nn.Linear(latent_dim, post_latent_dim)
        self.conv_net = ConvTransposeNet(net_arch, last_activation=last_activation)
        self.latent_dim = latent_dim
        self.crop_func = crop_func
    def forward(self, batch_size=-1, noise=None, debug=False):
        if noise is None:
            latent = th.randn(batch_size, self.latent_dim)
            noise = self.project_latent(latent)
            noise = noise.view(batch_size, *self.post_latent_shape)
        out = self.conv_net(noise, debug)
        out = self.crop_func(out)
        return out   
#%%
x = "nothing"
classifier_conv_arch = [
    #stride, pad before transform, out channels
    [x, x, 1], #dummy
    [2, 0, 32],
    [2, 1, 64],
    [1, 0, 64],
    [2, 0, 64]
]
classifier_mlp_arch = [64*16, 10, 1]

generator_arch = [
    [x, x, 64],
    [2, 0, 64],
    [2, 0, 32],
    [2, 1, 16],
    [1, 1, 1]
]
discriminator = ConvClassifier(classifier_conv_arch, classifier_mlp_arch, last_activation=lambda x: th.sigmoid(x))
generator = ConvGenerator(16, (64, 4, 4), generator_arch, crop_func=lambda x: x[:, :, :28, :28], last_activation=th.sigmoid, noise_device=device)
dopt = th.optim.Adam(discriminator.parameters(), 1e-4)
gopt = th.optim.Adam(generator.parameters(),     3e-4)
generator, discriminator = generator.to(device), discriminator.to(device)
#%%
for img, _ in train_loader:
    # res = discriminator(img, True)
    break
gen = generator(batch_size=3, debug=1)

plt.imshow(gen.detach().numpy()[0, 0]), discriminator(gen, debug=1)
# %%
writer = SummaryWriter()
nepoch = 20
epsilon = 1e-9
for ep in range(0, nepoch):
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        gen_img = generator(batch_size=img.shape[0])
        pred_fake = discriminator(gen_img)
        pred_real = discriminator(img)
        loss_generator = -th.log(pred_fake+epsilon).mean()
        loss_discriminator = -th.log(pred_real+epsilon).mean()/2 -th.log(1-pred_fake+epsilon).mean()/2

        if i%4!=0: #loss_generator.item() > loss_discriminator.item():
            gopt.zero_grad()
            loss_generator.backward()
            gopt.step()
        else:
            dopt.zero_grad()
            loss_discriminator.backward()
            dopt.step()

        writer.add_scalar('loss/generator', loss_generator, i+ep*len(train_loader))
        writer.add_scalar('loss/discriminator', loss_discriminator, i+ep*len(train_loader))
        if i % 30 == 0:
            print(f"{ep :2d}  {i:5d}  {loss_generator:7.5f}  {loss_discriminator:7.5f}")
            gen = generator(batch_size=1)
            writer.add_image('generator/sample', gen.detach()[0], i+ep*len(train_loader))

#
#%%
latent = th.tensor([[-0.1, -0.9]]).to(device)
pimg, _ = model(latent, all=False, std_randn=0.0)
toplot = pimg.cpu().detach().numpy()[0, 0, :28, :28]
plt.imshow(toplot)

# %%
vals = []
for i in range(10):
    ri = random.randint(0, len(dataset2)-1)
    sample = dataset2[ri][0].unsqueeze(0).to(device)
    _, (mu, std) = model(sample)
    vals.append((mu, std))
#%%
## morph between two random
import cv2
niter = 1000
a = vals[random.randint(0, len(vals)-1)]
b = vals[random.randint(0, len(vals)-1)]
for i in range(niter):
    alpha = i/niter
    mu = a[0]*alpha + b[0]*(1-alpha)
    pimg = model(mu, all=False, std_randn=0.0)[0].cpu().detach().numpy()
    cv2.imshow('slika', pimg[0, 0])
    key = cv2.waitKey(1)
cv2.destroyAllWindows()

# %%
# loop morph

niter = 500
frames = []
for iitem, (a, b) in enumerate(zip(vals, vals[1:]+vals[:1])):
    for i in range(niter):
        alpha = 1-i/niter #if iitem%2 else 1-i/niter
        mu = a[0]*alpha + b[0]*(1-alpha)
        pimg = th.sigmoid(model.decoder(mu)).view(28, 28, 1).detach().numpy()
        # cv2.imshow('slika', pimg)
        # key = cv2.waitKey(1)
        frames.append(pimg)
cv2.destroyAllWindows()
# %%
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(random.random())+".mp4", fourcc, 30, (28, 28))
#and write your frames in a loop if you want
for i, frame in enumerate(frames):
    out.write((frame*255).astype(np.uint8))
# %%
import numpy as np
import cv2
size = 28, 28
duration = 2
fps = 170
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for frame in frames:
    data = np.random.randint(0, 256, size, dtype='uint8')
    out.write((frame*255).astype(np.uint8))
out.release()


# %%

# %%
model.encoder.layers
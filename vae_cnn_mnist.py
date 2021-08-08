#%%
import numpy as np
import torch as th
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from models import ConvVAE
import matplotlib.pyplot as plt
from utils import loss_function_vae


transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('../data', train=False,               transform=transform)

train_loader = th.utils.data.DataLoader(dataset1, batch_size=32, shuffle=False)
test_loader = th.utils.data.DataLoader(dataset2)

device = th.device('cuda')

#%%
x = "nothing"
encoder_arch = [
    #stride, pad before transform, out channels
    [x, x, 1], #dummy
    [2, 0, 32],
    # [1, 0, 32],
    [2, 0, 64],
    [1, 0, 64],
    [2, 0, 32],
    # [1, 0, 32]
]
decoder_arch = [
    [x, x, 32],
    [2, 0, 32],
    [2, 0, 32],
    [2, 1, 16],
    [1, 1, 1]
]
model = ConvVAE(encoder_arch, decoder_arch, pre_z_shape=(32, 4, 4), z_size=2, post_z_shape=(32, 4, 4)).to(device)
model.load_state_dict(th.load("model_cnn_vae_2.pt"))
opt = th.optim.Adam(model.parameters(), lr=3e-4)

# %%
writer = SummaryWriter()
nepoch = 20
epsilon = 1e-9
for ep in range(0, nepoch):
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        pred_img, (m, log_std) = model(img)

        loss, (loss_like, loss_kl) = loss_function_vae(pred_img[:, :, :28, :28], img, m, log_std, 0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('loss/all', loss, i+ep*len(train_loader))
        writer.add_scalar('loss/likelihood', loss_like, i+ep*len(train_loader))
        writer.add_scalar('loss/kl', loss_kl, i+ep*len(train_loader))
        if i % 100 == 0:
            print(f"{ep :2d} {i:5d}  {loss:7.5f}  {loss_like:7.5f}  {loss_kl:7.5f}")

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
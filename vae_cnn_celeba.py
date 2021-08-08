#%%
import numpy as np
import torch as th
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from models import ConvVAE
from utils import loss_function_MSE_vae
import matplotlib.pyplot as plt

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
        ])
detransform = transforms.Compose([
        transforms.Normalize(mean = [ 0.000, 0.000, 0.000 ], std = [1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1, 1, 1 ])
])
        
dataset1 = datasets.ImageFolder('celeba', transform=transform)
dataset2 = datasets.ImageFolder('celeba', transform=transform)

train_loader = th.utils.data.DataLoader(dataset1, batch_size=12, shuffle=True)
test_loader = th.utils.data.DataLoader(dataset2)

device = th.device('cuda')
#%%
for batch in train_loader:
    break
tensors2img = lambda x: detransform(x[0][0].cpu()).transpose(0, -1).transpose(-3, -2).numpy()
plt.imshow(tensors2img(batch))
#%%
from utils import save, load
load(model, "celeba_vae_100_model", 0, 0)
load(opt, "celeba_vae_100_opt", 0, 0)
#%%
x = "nothing"
encoder_arch = [
    #pool, pad before transform, out channels
    [x, x, 3], #dummy
    [2, 0, 32],
    [2, 0, 32],
    [2, 0, 64],
    [1, 0, 64],
    [2, 0, 128],
    # [1, 0, 32]
]
decoder_arch = [
    #stride, pad before transform, out channels
    [x, x, 128],
    [2, 0, 128],
    [2, 0, 64],
    [2, 1, 32],
    [2, 1, 16],
    [1, 1, 3]
]
try:
    del batch
    del img
    del model
except:
    pass
model = ConvVAE(encoder_arch, decoder_arch, pre_z_shape=(128, 14, 12), z_size=100, post_z_shape=(128, 14, 12), last_activation=lambda x: x).to(device)
# model.load_state_dict(th.load("model_cnn_vae_2.pt"))
#%%
opt = th.optim.Adam(model.parameters(), lr=3e-5)

#%%
out = model(batch[0].to(device), debug=True)
out[0].shape
# %%
try:
    del img
    del pred_img
    del loss
    del loss_like
    del loss_kl
except:
    print("nothing to del")
writer = SummaryWriter()
nepoch = 20
epsilon = 1e-9
model = model.to(device)
for ep in range(0, nepoch):
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        pred_img, (m, log_std) = model(img)

        loss, (loss_like, loss_kl) = loss_function_MSE_vae(pred_img[:, :, :218, :178], img, m, log_std, 0.001)

        print(f"{ep :2d} {i:5d}  {loss:7.5f}  {loss_like:7.5f}  {loss_kl:7.5f}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('loss/all', loss, i+ep*len(train_loader))
        writer.add_scalar('loss/likelihood', loss_like, i+ep*len(train_loader))
        writer.add_scalar('loss/kl', loss_kl, i+ep*len(train_loader))
        if i % 100 == 0:
            print(f"{ep :2d} {i:5d}  {loss:7.5f}  {loss_like:7.5f}  {loss_kl:7.5f}")
        if i%2000 == 0:
            save(model, "celeba_vae_100_model", ep, i)
            save(opt, "celeba_vae_100_opt", ep, i)


#%%
#%%
latent = th.tensor([[-0.1, -0.9]]).to(device)
pimg, _ = model(latent, all=False, std_randn=0.0)
toplot = pimg.cpu().detach().numpy()[0, 0, :28, :28]
plt.imshow(toplot)

# %%3
from utils import plot_sampled_x_and_gen
import random
def plot_sampled_x_and_gen(model, dataset, device=th.device('cpu'),  dataset2img = lambda x: x[0], detransform= lambda x:x):
    randi = random.randint(0, len(dataset)-1)
    img = dataset2img( dataset[randi] )
    img = img.unsqueeze(0).to(device)
    pimg, other = model.to(device)(img, std_randn=0)
    pimg, img = detransform(pimg[0].cpu()), detransform(img[0].cpu())
    toplot = th.cat((img, pimg[:, :img.shape[-2], :img.shape[-1]]), dim=2).detach().numpy()
    if toplot.shape[0] == 3:
        toplot = toplot.transpose(1, 2, 0)
    plt.imshow(toplot)
    return toplot, other 
_ = plot_sampled_x_and_gen(model, dataset1, detransform=detransform)
#%%
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
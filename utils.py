import torch.nn.functional as F
import torch as th
import random
import matplotlib.pyplot as plt
import os


def loss_function_BCE_vae(recon_x, x, mu, log_std, beta=1):
    log_var = 2*th.abs(log_std)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')/x.shape[0]
    KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD*beta, (BCE, KLD)


def loss_function_MSE_vae(recon_x, x, mu, log_std, beta=1):
    log_var = 2*th.abs(log_std)
    BCE = ((recon_x-x)**2).sum()/x.shape[0]
    KLD = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD*beta, (BCE, KLD)


def plot_sampled_x_and_gen(model, dataset, device=th.device('cpu'),  dataset2img = lambda x: x[0], detransform= lambda x:x):
    randi = random.randint(0, len(dataset)-1)
    img = dataset2img( dataset[randi] )
    img = img.unsqueeze(0).to(device)
    pimg, other = model.to(device)(img, std_randn=0)
    pimg, img = detransform(pimg[0].cpu()), img[0].cpu()
    toplot = th.cat((img, pimg[:, :img.shape[-2], :img.shape[-1]]), dim=1).detach().numpy()
    plt.imshow(toplot)
    return toplot, other 


def save(model, name, major, minor, dir='pretrained_models/'):
    dir_create(dir)
    PATH =  os.path.join(dir, name+'_'+str(major)+'_'+str(minor)+'.th')
    th.save(model.state_dict(), PATH)
def load(model, name, major, minor, dir='pretrained_models/'):
    dir_create(dir)
    PATH =  os.path.join(dir, name+'_'+str(major)+'_'+str(minor)+'.th')
    model.load_state_dict(th.load(PATH))
def dir_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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




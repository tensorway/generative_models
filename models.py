import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, net_arch, last_activation = lambda x: x):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.last_activation = last_activation
    def forward(self, x):
        h = x.view(x.shape[0], -1)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return self.last_activation(h)

class MLP_VAE(nn.Module):
    def __init__(self, encoder_arch, decoder_arch):
        super().__init__()
        self.encoder = MLP(encoder_arch)
        self.decoder = MLP(decoder_arch)
        self.mean_project = nn.Linear(encoder_arch[-1], decoder_arch[0])
        self.std_project  = nn.Linear(encoder_arch[-1], decoder_arch[0])
    def forward(self, x):
        h = self.encoder(x)
        m = self.mean_project(h)
        std = th.exp(self.std_project(h)/2)
        latent = m + std*th.randn_like(std)
        h = self.decoder(latent)
        h = th.sigmoid(h)
        return h.view(x.shape), (m, std)

class ConvResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, route_to):
        super().__init__()
        self.normal = True
        if in_channels != out_channels or stride != 1:
            self.normal = False 
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.route_to = route_to
    def forward(self, x):
        if self.normal:
            return x + self.route_to(x)
        return self.conv(x) + self.route_to(x)

class ConvStrideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.pad = nn.ZeroPad2d(1)
    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        h = F.relu(h)
        h = self.bnorm(h)
        return h   

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(pool_size)
        self.pad = nn.ZeroPad2d(1)
    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        h = self.pool(h)
        h = F.relu(h)
        h = self.bnorm(h)
        return h 

class ConvNet(nn.Module):
    def __init__(self, net_arch, last_activation = lambda x: x):
        super().__init__()
        layers = []
        for (_, _, in_channels), (stride, pad, out_channels) in zip(net_arch[:-1], net_arch[1:]):
            conv = ConvBlock(in_channels, out_channels, stride)
            if pad != 0:
                layers.append(nn.ZeroPad2d(pad))
            layers.append(ConvResidualConnection(in_channels, out_channels, stride, conv))
        self.layers = nn.ModuleList(layers)
        self.last_activation = last_activation
    def forward(self, x, debug=False):
        h = x
        for i, lay in enumerate(self.layers[:-1]):
            if debug:
                print(i, h.shape)
            h = lay(h)
        h = self.layers[-1](h)
        if debug:
            print("last shape from convnet", h.shape)
        return self.last_activation(h)



# TODO make merge transpose and normal
class ConvResidualTransposeConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, route_to):
        super().__init__()
        self.normal = True
        if in_channels != out_channels or stride != 1:
            self.normal = False 
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride)#, padding=1)
        self.route_to = route_to
    def forward(self, x):
        if self.normal:
            return x + self.route_to(x)
        return self.conv(x) + self.route_to(x) #self.conv(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bnorm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bnorm(h)
        return h

class ConvTransposeNet(nn.Module):
    def __init__(self, net_arch, last_activation = lambda x: x):
        super().__init__()
        layers = []
        for (_, _, in_channels), (stride, pad, out_channels) in zip(net_arch[:-1], net_arch[1:]):
            if pad != 0:
                layers.append(nn.ZeroPad2d(pad))
            conv = ConvTransposeBlock(in_channels, out_channels, stride)
            layers.append(ConvResidualTransposeConnection(in_channels, out_channels, stride, conv))
        self.layers = nn.ModuleList(layers)
        self.last_activation = last_activation
    def forward(self, x, debug=False):
        h = x
        for i, lay in enumerate(self.layers[:-1]):
            if debug:
                print(i, h.shape)
            h = lay(h)
        h = self.layers[-1](h)
        if debug:
            print(h.shape, "last shape from transposenet")
        return self.last_activation(h)


class LatentSampler(nn.Module):
    def __init__(self, pre_shape, z_size, post_shape):
        super().__init__()
        mul = 1
        pre_z_size = [mul:=mul*item for item in pre_shape][-1]        
        mul = 1
        post_z_size = [mul:=mul*item for item in post_shape][-1]
        self.mean_project = nn.Linear(pre_z_size, z_size)
        self.std_project  = nn.Linear(pre_z_size, z_size)
        self.de_project   = nn.Linear(z_size, post_z_size)
        self.post_shape = post_shape
    def forward(self, h, std_randn=1, all=True):
        bsize = h.shape[0]
        if all:
            h = h.view(bsize, -1)
            m = self.mean_project(h)
            log_std = self.std_project(h)
            std = th.exp(log_std)
            latent = m + std*th.randn_like(std)*std_randn
        else:
            latent, (m, log_std) = h, (h, -1)
        h = self.de_project(latent).view(bsize, *self.post_shape)
        return h, (m, log_std)


class ConvVAE(nn.Module):
    def __init__(self, encoder_arch, decoder_arch, pre_z_shape, z_size, post_z_shape, last_activation = lambda x: th.sigmoid(x)):
        super().__init__()
        self.encoder = ConvNet(encoder_arch)
        self.decoder = ConvTransposeNet(decoder_arch,  last_activation=last_activation)
        self.sampler = LatentSampler(pre_z_shape, z_size, post_z_shape)

    def forward(self, h, std_randn=1, all=True, debug=False):
        if all:
            h = self.encoder(h, debug=debug)
        latent, (m, log_std) = self.sampler(h, std_randn, all=all)
        h = self.decoder(latent, debug=debug)
        return h, (m, log_std)
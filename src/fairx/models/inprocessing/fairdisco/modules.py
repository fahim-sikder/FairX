#This code is adapted from https://github.com/SoftWiser-group/FairDisCo/tree/main

from .utils import *
import math
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

import warnings
import os
warnings.filterwarnings('ignore')

class FairDisCo(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, s_dim, d):
        super(FairDisCo, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.d = d

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + s_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 2 * z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + s_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, x_dim),
        )

    def encode(self, x, s):
        # P(Z|X,S)
        s = F.one_hot(s, self.s_dim)
        x = torch.cat([x, s], dim=1)
        h = self.encoder(x)
        self.mean = h[:,:self.z_dim]
        self.logvar = h[:,self.z_dim:]
        self.var = torch.exp(self.logvar)
        # reparametrize
        gaussian_noise = torch.randn_like(self.mean, device=self.mean.device)
        z = self.mean + gaussian_noise * torch.sqrt(self.var)
        return z

    def decode(self, z, s):
        # P(X|Z,S)
        s = F.one_hot(s, self.s_dim)
        z = torch.cat([z, s], dim=1)
        return self.decoder(z)

    def dis(self, index_i, index_j):
        var_i, var_j = self.var[index_i], self.var[index_j]
        mean_i, mean_j = self.mean[index_i], self.mean[index_j]
        item1 = var_i.unsqueeze(1) + var_j
        item2 = (mean_i.unsqueeze(1) - mean_j)**2 / item1
        item2 = torch.exp(-item2.sum(-1) / 2)
        item3 = torch.sqrt((2*math.pi*item1).prod(-1)) + 1e-10
        ans = (item2 / item3).mean()
        return ans

    def calculate_dis(self, s, batch_size):
        # V^2(p(Z,S),p(S)p(Z))
        ans = 0
        num = s.shape[0]
        index_j = s>-1
        item1 = self.dis(index_j, index_j)
        for i in range(self.s_dim):
            index_i = s==i
            num_i = index_i.sum()
            if 0 < num_i < num:
                cur = item1 + self.dis(index_i, index_i) - 2 * self.dis(index_i, index_j)
                ans += (num_i / num)**2 * cur
        ans *= num / batch_size
        return ans

    def calculate_kl(self):
        # kl
        return -0.5 * (1 + self.logvar - self.mean**2 - self.var).sum(dim=-1).mean()

    def calculate_re(self, x_hat, c):
        # l(X_hat, X)
        ans = 0
        left, right = 0, 0
        for i, length in enumerate(self.d):
            right = left + length
            ans += F.cross_entropy(x_hat[:, left:right], c[:, i])
            left = right
        return ans

    def fit(self, train_data, epochs, lr, batch_size, verbose, beta, device):
        assert beta >= 0
        self.to(device=device)
        self.train()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(1, epochs+1)):
            train_loss = 0
            num = 0
            # train_step
            for x, c, s, y in train_loader:
                x = x.to(device)
                c = c.to(device)
                s = s.to(device)
                y = y.to(device)

                z = self.encode(x, s)
                out = self.decode(z, s)
                # loss
                re_loss = self.calculate_re(out, c)
                Dis_loss = self.calculate_dis(s, batch_size) if beta > 1 else 0
                kl_loss = self.calculate_kl()

                loss = re_loss + kl_loss + beta * Dis_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # eval
                train_loss += loss.detach() * x.shape[0]
                num += x.shape[0]
            
            train_loss = (train_loss / num).item()
            if verbose > 0 and epoch % verbose == 0:
                print('Epoch {:04d}: train_loss={:.5f}'.format(epoch, train_loss))
        self.to('cpu')
        self.eval()

    def load(self, path):
        self.load_state_dict(torch.load(path))

class FairDisCoImage(nn.Module):
    def __init__(self, z_dim, s_dim, n_chan=3, im_shape=(64, 64)):
        super(FairDisCoImage, self).__init__()
        self.im_shape = im_shape
        self.n_chan = n_chan
        self.z_dim = z_dim
        self.s_dim = s_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_chan, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + s_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, n_chan, 4, 2, 1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        # P(Z|X,S)
        h = self.encoder(x)
        self.mean = h[:,:self.z_dim]
        self.logvar = h[:,self.z_dim:]
        self.var = torch.exp(self.logvar)
        # reparametrize
        gaussian_noise = torch.randn_like(self.mean, device=x.device)
        z = self.mean + gaussian_noise * torch.sqrt(self.var)
        return z

    def decode(self, z, s):
        # P(X,S|Z,S)
        s = F.one_hot(s, self.s_dim)
        z = torch.cat([z, s],dim=1)
        z = z.view(-1,self.z_dim+self.s_dim,1,1)
        x_hat = self.decoder(z)
        return x_hat
    
    def calculate_kl(self):
        # kl
        return -0.5 * (1+self.logvar-self.mean**2-torch.exp(self.logvar)).sum() / self.mean.shape[0]
    
    def calculate_re(self, x_hat, x):
        # loss(x_hat, x)
        return F.binary_cross_entropy(x_hat, x, reduction='sum') / x.shape[0]

    def dis(self, index_i, index_j):
        var_i, var_j = self.var[index_i], self.var[index_j]
        mean_i, mean_j = self.mean[index_i], self.mean[index_j]
        item1 = var_i.unsqueeze(1) + var_j
        item2 = (mean_i.unsqueeze(1) - mean_j)**2 / item1
        item2 = torch.exp(-item2.sum(-1) / 2)
        item3 = torch.sqrt((2*math.pi*item1).prod(-1)) + 1e-10
        ans = (item2 / item3).mean()
        return ans

    def calculate_dis(self, s, batch_size):
        # V^2(p(z,s),p(s)p(z))
        ans = 0
        num = s.shape[0]
        index_j = s>-1
        item1 = self.dis(index_j, index_j)
        for i in range(self.s_dim):
            index_i = s==i
            num_i = index_i.sum()
            if 0 < num_i < num:
                cur = item1 + self.dis(index_i, index_i) - 2 * self.dis(index_i, index_j)
                ans += (num_i / num)**2 * cur
        ans *= num / batch_size
        return ans
    
    def fit(self, train_data, epochs, lr, batch_size, verbose, beta, device):
        self.to(device=device)
        self.train()
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            train_loss = 0
            num = 0
            # train_step
            for x, s, _ in train_loader:
                x = x.to(device)
                s = s.to(device)
                z = self.encode(x)
                x_hat = self.decode(z, s)
                # loss
                re_loss = self.calculate_re(x_hat, x)
                Dis_loss = self.calculate_dis(s, batch_size) if beta > 1 else 0
                kl_loss = self.calculate_kl()

                loss = re_loss + kl_loss + beta * Dis_loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # eval
                train_loss += loss.detach() * x.shape[0]
                num += x.shape[0]

            train_loss = (train_loss / num).item()
            if verbose > 0 and epoch % verbose == 0:
                print('Epoch {:04d}: train_loss={:.5f}'.format(epoch, train_loss))
        self.to('cpu')
        self.eval()

    def load(self, path):
        self.load_state_dict(torch.load(path))
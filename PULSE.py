import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import os
import json
import gdown

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from PULSE.loss.sphericaloptimizer import SphericalOptimizer

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

class PULSE(object):

    def __init__(self, data='processed images/', batch_size=64, image_size=32, lr=0.0001, ngpu=1):

      #path to the dataset used for training which has preprcossed downsampled images.
      self.dataroot = data

      #the batch size used in training.
      self.batch_size = batch_size
      
      #the spatial size of the image used for training. 
      self.image_size = image_size

      #learning rate for training.
      self.lr = lr

      #number of GPUs available for training. If no GPU is available, the model will train on CPU. Here, we have only 1 GPU available.
      self.ngpu = ngpu

      if ngpu > 0 and not torch.cuda.is_available():
        raise ValueError('ngpu > 0 but cuda not available')

      #device used for training.
      self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

      #linear mapping layer used to map the latent distribution to that of the input mapping network
      self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

      #the generator network of the stylegan
      self.synthesis = G_synthesis().cuda()

      #the input mapping network of the stylegan
      self.inp_mapping = G_mapping().cuda()

      if ngpu > 0 and not torch.cuda.is_available():
        raise ValueError('ngpu > 0 but cuda not available')

      print('Loading the synthesis network')

      #download weights for the pre-trained generator network of the stylegan
      with open( __PREFIX__+"/config/file_downloader.json", 'rb') as fp:
        json_file = json.load(fp)
        url = 'https://drive.google.com/uc?id={}'.format(json_file['synthesis'])
        gdown.download(url, 'synthesis.pt', quiet=False)
        f1 = 'synthesis.pt'
        self.synthesis.load_state_dict(torch.load(f1))

      for params in self.synthesis.parameters():
          params.requires_grad = False

      print('Loading the input mapping network')

      #download weights for the pre-trained input mapping network of the stylegan
      with open( __PREFIX__+"/config/file_downloader.json", 'rb') as fp:
        json_file = json.load(fp)
        url = 'https://drive.google.com/uc?id={}'.format(json_file['mapping'])
        gdown.download(url, 'mapping.pt', quiet=False)
        f1 = 'mapping.pt'
        self.inp_mapping.load_state_dict(torch.load(f1))

      #create a gaussian distribution of latent vectors
      with torch.no_grad():

        latent_input = torch.randn((1000000,512), dtype=torch.float32, device="cuda")
        latent_output = torch.nn.LeakyReLU(5)(inp_mapping(latent_input))
        self.gaussian = {"mean": latent_output.mean(0), "std": latent_output.std(0)}
    
        #save the distribution as a pytorch file.
        torch.save(self.gaussian, "gaussian.pth")
    
    def data_loader(self):

        #create the dataset
        dataset = dset.ImageFolder(root = self.dataroot,
                                transform = transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.CenterCrop(self.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

        #create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,
                                                shuffle = True)

        return dataloader
      
    def train(self):

        latent = torch.randn((self.batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        dataloader = self.data_loader()

        #generate a list of noise tensors
        noise = [] # stores all of the noise tensors
        #noise_optimizer = []  # stores the noise tensors that we want to optimize on

        for i in range(18):

            # dimension of the ith noise tensor
            res = (self.batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            #generate a random tensor that is to be used as noise
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            new_noise.requires_grad = True

            #append the noise tensors in a list
            noise.append(new_noise)

        #add the noise to the latent distribution
        vars = [latent] + noise

        #set up Adam as the base optimizer function
        optimizer_function = optim.Adam

        #modify the adam optimizer to work for hyperspheres
        optimizer = SphericalOptimizer(optimizer_function, vars, self.lr)
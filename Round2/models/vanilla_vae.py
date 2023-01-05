import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from keras.backend import shape
import matplotlib.pyplot as plt
import os
import numpy as np

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim), #, track_running_stats=False),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4**2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4**2, latent_dim)
        self.fc_TwoLayer_1 = nn.Linear(latent_dim,512)
        self.fc_TwoLayer_2 = nn.Linear(512,256)
        self.fc_threeLayer_3 = nn.Linear(256,1)

        for param in self.encoder.parameters():
          param.requires_grad = False
        for param in self.fc_mu.parameters():
          param.requires_grad = False
        for param in self.fc_var.parameters():
          param.requires_grad = False
        # for param in self.fc_TwoLayer_1.parameters():
        #   param.requires_grad = False
        # for param in self.fc_TwoLayer_2.parameters():
        #   param.requires_grad = False
        # for param in self.fc_threeLayer_3.parameters():
        #   param.requires_grad = False
        # Build 2-layer FC
        # modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *4)

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )



        # self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 13,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input[:,0:13,:,:])
        # print(input[:,0:13,:,:])
        # result = result.view(2048,-1)
        result = torch.flatten(result, start_dim=1)
        # print(result)
        # print(shape(result.cpu()))
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D] %%%%%%%%% changed to mu: (Tensor) [B x D] 
        
        :return: (Tensor) [B x C x H x W]
        """
        # result = self.encode()
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2,2)
        result = self.decoder(result)
        result = self.final_layer(result)


        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        real_img_LiDAR_0 = input[:,0:13,:,:]
        real_img_LST_0 = input[:,13,:,:]  # 300, (:,128,128)
        # real_img_LST = np.concatenate((real_img_LST_0[:,:,0:64,0:64],real_img_LST_0[:,:,0:64,64:128],real_img_LST_0[:,:,64:128,0:64],real_img_LST_0[:,:,64:128,64:128]),axis = 0)
        # real_img_LiDAR = np.concatenate((real_img_LiDAR_0[:,:,0:64,0:64],real_img_LiDAR_0[:,:,0:64,64:128],real_img_LiDAR_0[:,:,64:128,0:64],real_img_LiDAR_0[:,:,64:128,64:128]),axis = 0)
        mu, log_var = self.encode(real_img_LiDAR_0)
        # print('mu',mu,'*******','log_var',log_var[0],'****************')


        # mu_1 = mu.reshape(4,-1,1024)
        # mu = np.concatenate((mu_1[0,:,:],mu_1[1,:,:],mu_1[2,:,:],mu_1[3,:,:]),axis = 1) 
        result = self.fc_TwoLayer_1(mu)
        result = nn.functional.leaky_relu(result)
        result = self.fc_TwoLayer_2(result)
        result = nn.functional.leaky_relu(result)
        result = self.fc_threeLayer_3(result)
        result_kalvin = result*60.+270  # 0.1--->310 (:,1)
        lst_kalvin = real_img_LST_0   # 300, (:,128,128)
        a = real_img_LST_0.cpu().detach().numpy()
        lst_norm = np.sum((a-270)/60.,axis=(1,2))/(128*128.)  # 0.1 (:,128,128) --> (:,1)
        # lst_norm = (lst_norm-270)/60.
        # print(lst_norm)
        
        # print('result',result)
        return  [result,result_kalvin, lst_kalvin, lst_norm ]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        # input = args[1]
        lst = args[2]

        
        lst1 = lst.cpu().detach().numpy()
        lst = (lst1-270)/60.
        lst1 = np.sum(lst1,axis=(1,2))/(128*128.)      ################ 128 ---> 256
        lst1 = lst1.reshape(-1,1)
        lst1 = torch.from_numpy(lst1)
        lst1 = lst1.to(0)
        lst = np.sum(lst,axis = (1,2))/(128*128.)
        lst = lst.reshape(-1,1)
        # print(lst.shape)
        lst = torch.from_numpy(lst)
        lst = lst.to(0)
        # print('recons',recons,'lst',lst)
        
        
        # print('recons_loss',recons_loss)
        # print('normalized_after loss',recons*60.+270)
        # print('lst1',lst1)
        # print('diff_',lst1-(recons*60.+270))

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # with torch.no_grad():
        recons_loss = F.mse_loss(recons, lst)
        loss = recons_loss # + kld_weight * kld_loss
        # print(recons, lst,loss)
        c = recons*60.+270
        # c = c.cpu().detach().numpy()
        Diff = F.mse_loss(lst1, c)
        # print('MSE',Diff)
        # print('Diff',lst1.reshape(1,-1)-c.reshape(1,-1))
        # print('loss',loss)
        recons = recons.cpu().detach().numpy()
        # recons_los = F.mse_loss(recons, input)#,reduction = 'sum') #/torch.sum(mu**2,dim=1)
        # recons_los = nn.functional.binary_cross_entropy(recons, input, reduction='sum')
        # i = input.cpu().detach().numpy()
        # np.save("/content/sample_data/b.npy",i)
        # a = recons.cpu().detach().numpy()
        # np.save("/content/sample_data/a.npy",a)

        return {'loss': loss,'Diff':Diff } #'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        # # Display image and label.
        # train_features = next(iter(self.forward(x)[0]))
        # print(f"Feature batch shape: {train_features.size()}")
        # # print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0].squeeze()
        # # label = train_labels[0]
        # plt.imshow(img, cmap="gray")
        # plt.savefig('/content/drive/MyDrive/TUM/PyTorch-VAE/a.jpg')

        return self.forward(x)[0],self.forward(x)[1]
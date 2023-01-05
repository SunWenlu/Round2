import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.requires_grad = False
        self.automatic_optimization = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img= batch
        self.curr_device = real_img.device

        opt = self.optimizers()
        self.model.zero_grad()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optimizer.zero_grad()
        results = self.forward(real_img)        
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        # for p in self.model.parameters():
        #   p.requires_grad = False

        # self.manual_backward(results)
        self.log_dict({key: val for key, val in train_loss.items()}, sync_dist=True)

        lst_norm = results[3]
        lst_kelvin = results[2].cpu().detach().numpy()
        lst_kelvin = np.sum(lst_kelvin,axis=(1,2))/(128*128.)
        generated_norm = results[0]
        generated_kelvin = results[1].cpu().detach().numpy()
        lst_kelvin = lst_kelvin.reshape(-1,1)

        # ((40*n_epoch + batch_idx-1))*5e-9
        iteration = self.current_epoch +batch_idx

        x=np.arange(1,101).reshape(100,1)
        l1=plt.plot(x,generated_kelvin[0:100],'r--',label='predicted temperature')
        l2=plt.plot(x,lst_kelvin[0:100],'b--',label='ground truth')
        # l3=plt.plot(x,y3,'b--',label='type3')
        plt.plot(x,generated_kelvin[0:100],'ro-',x,lst_kelvin[0:100],'b+-',)
        plt.title('Training: The Predicted Temperature and Ground Truth in Kelvin')
        plt.xlabel('sample')
        plt.ylabel('temperature')
        plt.legend()
        plt.savefig('/content/sample_data/curve_training.png')
        plt.show()

        Diff = generated_kelvin-lst_kelvin
        print('Diff',Diff.shape,lst_kelvin.shape,generated_kelvin.shape,Diff)
        plt.plot(x,Diff[0:100],'b--')
        plt.title('Training: The Predicted Temperature - Ground Truth in Kelvin')
        plt.xlabel('Temperature (Kelvin)')
        plt.ylabel('PT - GT')
        # plt.text(330, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(270, 330)
        # plt.ylim(0, 0.03)
        plt.grid(True)
        plt.savefig('/content/sample_data/PT-GT_training.png')
        plt.show()
        # Diff = generated_kelvin-lst_kelvin
        # print('Diff',Diff.shape,lst_kelvin.shape,generated_kelvin.shape,Diff)
        n, bins, patches = plt.hist(Diff[0:6000], 30,  facecolor='g', alpha=0.75)
        plt.xlabel('Temperature (Kelvin)')
        plt.ylabel('Count')
        plt.title('Histogram of Diff ')
        # plt.text(330, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(270, 330)
        # plt.ylim(0, 0.03)
        plt.grid(True)
        plt.savefig('/content/sample_data/Diff_histogram_training'+str(iteration)+'.png')
        plt.show()


        n, bins, patches = plt.hist(generated_kelvin[0:6000], 30,  facecolor='g', alpha=0.75)
        plt.xlabel('Temperature (Kelvin)')
        plt.ylabel('Count')
        plt.title('Histogram of Diff ')
        # plt.text(330, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(270, 330)
        # plt.ylim(0, 0.03)
        plt.grid(True)
        plt.savefig('/content/sample_data/lst_histogram_training.png')
        plt.show()


        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device

        iteration = self.current_epoch +batch_idx

        opt = self.optimizers()
        self.model.zero_grad()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optimizer.zero_grad()

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
                                           
        # [result,result_kalvin, lst_kalvin, lst_norm ]
        # self.manual_backward(results)

        lst_norm = results[3]
        lst_kelvin = results[2].cpu().detach().numpy()
        lst_kelvin = np.sum(lst_kelvin,axis=(1,2))/(128*128.)
        generated_norm = results[0]
        generated_kelvin = results[1].cpu().detach().numpy()
        lst_kelvin = lst_kelvin.reshape(-1,1)

        if  ((self.current_epoch % 10) == 0) or (self.current_epoch == 99):

          # the histogram of the data
          # n, bins, patches = plt.hist(generated_kelvin, 50, density=True, facecolor='g', alpha=0.75)



          x=np.arange(1,21).reshape(20,1)
          l1=plt.plot(x,generated_kelvin[0:20],'r--',label='predicted temperature')
          l2=plt.plot(x,lst_kelvin[0:20],'b--',label='ground truth')
          # l3=plt.plot(x,y3,'b--',label='type3')
          plt.plot(x,generated_kelvin[0:20],'ro-',x,lst_kelvin[0:20],'b+-',)
          plt.title('The Predicted Temperature and Ground Truth in Kelvin')
          plt.xlabel('sample')
          plt.ylabel('temperature')
          plt.legend()
          plt.savefig('/content/sample_data/curve.png')
          plt.show()

          Diff = generated_kelvin-lst_kelvin
          plt.plot(Diff[0:20],'b+-')
          plt.title('The Predicted Temperature - Ground Truth in Kelvin')
          plt.xlabel('Temperature (Kelvin)')
          plt.ylabel('PT - GT')
          # plt.text(330, .025, r'$\mu=100,\ \sigma=15$')
          # plt.xlim(270, 330)
          # plt.ylim(0, 0.03)
          plt.grid(True)
          plt.savefig('/content/sample_data/PT-GT.png')
          plt.show()
          
          n, bins, patches = plt.hist(Diff[0:600], 30,  facecolor='g', alpha=0.75)
          plt.xlabel('Temperature (Kelvin)')
          plt.ylabel('Count')
          plt.title('Histogram of Diff ')
          # plt.text(330, .025, r'$\mu=100,\ \sigma=15$')
          # plt.xlim(270, 330)
          # plt.ylim(0, 0.03)
          plt.grid(True)
          plt.savefig('/content/sample_data/Diff_histogram_validating'+str(iteration)+'.png')
          plt.show()

        # img = results[2]
        # columns = 2
        # # rows = 5
        
        #   for i in range(1, columns*rows +1):
        #     os.makedirs('/content/sample_data/Result/'+str(i),exist_ok=True)
        #     os.makedirs('/content/sample_data/Input/'+str(i),exist_ok=True)
        #     for j in range(13):
                
        #         img1 = img[i-1,j,:,:]
        #         # os.makedirs('/content/drive/MyDrive/TUM/Recon_0_',exist_ok=True)
        #         vutils.save_image(img1,
        #           os.path.join("/content/sample_data/Result/"+str(i)+"/"+str(j)+".png"),
        #           normalize=True,
        #           nrow=5)
        #         vutils.save_image(lst,
        #           os.path.join("/content/sample_data/Input/"+str(i)+"/"+str(j)+".png"),
        #           normalize=True,
        #           )
        #         img = inputs[i-1,j,:,:]
        #         vutils.save_image(img,
        #           os.path.join('/content/sample_data/Input/'+str(i)+"/"+str(j)+".png"),
        #           normalize=True,
        #           nrow=5)

        self.log_dict({f"val_{key}": val for key, val in val_loss.items()}, sync_dist=True)
        
       
        
    # def on_validation_end(self) -> None:
    #     self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)
        # print(test_input.shape)
#         test_input, test_label = batch
        recons,inputs = self.model.generate(test_input)
        # inputs = inputs[0,2,:,:]
        # print(recons.shape)
        if  ((self.current_epoch % 10) == 0) or (self.current_epoch == 399):
          for i in range(13):
              a = recons[0,i,:,:]
              b = inputs[0,i,:,:]
              
              vutils.save_image(a.data,
                                os.path.join(self.logger.log_dir , 
                                            "Reconstructions",
                                            f"recons_{self.logger.name}_Epoch_{self.current_epoch}_"+str(i)+".png"),
                                normalize=True,
                                nrow=1)
              vutils.save_image(b.data,
              os.path.join("/content/sample_data/a"+str(i)+"_.png"),
              normalize=True,
              nrow=1)

          # try:
              samples = self.model.sample(1,
                                          self.curr_device,
                                          )
              c = samples[0,i,:,:]
              vutils.save_image(c.cpu().data,
                                os.path.join(self.logger.log_dir , 
                                            "Samples",      
                                            f"{self.logger.name}_Epoch_{self.current_epoch}"+str(i)+".png"),
                                normalize=True,
                                nrow=1)
          # except Warning:
              # pass


    def configure_optimizers(self):

        optims = []
        scheds = []
        
        # param.grad = 0
        for param in self.model.encoder.parameters():
          param.requires_grad = False
        for param in self.model.fc_mu.parameters():
          param.requires_grad = False
        for param in self.model.fc_var.parameters():
          param.requires_grad = False
        optimizer = optim.Adam(self.model.parameters(),#filter(lambda p: p.requires_grad, self.model.parameters()),  #s
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # for name, param in self.model.named_parameters():
        #   if param.requires_grad:
        #     print(name, param.data)
        # for param in self.model.fc_TwoLayer_1.parameters():
        #   param.requires_grad = False
        # for param in self.model.fc_TwoLayer_2.parameters():
        #   param.requires_grad = False
        # for param in self.model.fc_threeLayer_3.parameters():
        #   param.requires_grad = False

        # Check if more than 1 optimizer is required (Used for adversarial training)
        # try:
        #     if self.params['LR_2'] is not None:
        #         optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
        #                                 lr=self.params['LR_2'])
        #         optims.append(optimizer2)
        # except:
        #     pass

        # try:
        #     if self.params['scheduler_gamma'] is not None:
        #         scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
        #                                                      gamma = self.params['scheduler_gamma'])
        #         scheds.append(scheduler)

        #         # Check if another scheduler is required for the second optimizer
        #         try:
        #             if self.params['scheduler_gamma_2'] is not None:
        #                 scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
        #                                                               gamma = self.params['scheduler_gamma_2'])
        #                 scheds.append(scheduler2)
        #         except:
        #             pass
        #         return optims, scheds
        # except:
        return optims

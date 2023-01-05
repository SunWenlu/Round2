import os
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
# torch.cuda.manual_seed_all(1)
model = vae_models[config['model_params']['name']](**config['model_params'])

Path = '/content/drive/MyDrive/TUM/PyTorch-VAE/model_JSC.pth'
# '/content/drive/MyDrive/TUM/PyTorch-VAE/model_2_stage_200bs.pth'   # 1002.pth' # manually resize from 64*64
# Path = '/content/drive/MyDrive/TUM/PyTorch-VAE/model.pth'
checkpoint = torch.load(Path)
# # add checkpoint into model
model_dict = model.state_dict()
# model.load_state_dict(checkpoint['model_state_dict'])
save_model = torch.load(Path)

model_dict = model.state_dict()
state_dict = { k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}

for key in model_dict.keys():
  print(key)
# print('**********************')
# print('save_model',save_model['model_state_dict'].keys())
# print('*************************************************',state_dict.keys())
model_dict.update(state_dict)
model.load_state_dict(model_dict)
# # for key in state_dict.keys():
# #   print(key)
# print('**********************')
# for key in model_dict.keys():
#   print(key)
# print('done!')
for param in model.encoder.parameters():
  param.requires_grad = False
for param in model.fc_mu.parameters():
  param.requires_grad = False
for param in model.fc_var.parameters():
  param.requires_grad = False

#################### 在测试的时候（！！！即加载已经训练好的模型时，需要添加下列代码：——————） 

for param in model.fc_TwoLayer_1.parameters():
  param.requires_grad = False
for param in model.fc_TwoLayer_2.parameters():
  param.requires_grad = False
for param in model.fc_threeLayer_3.parameters():
  param.requires_grad = False
 
# model.eval()
experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                #  strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


# Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")



# run trainer
a =list(model.parameters())[1]
# print('before_encoder',list(model.parameters()))
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
# print('before_fc',list(model.parameters())[20])
print("*")
#                                # 测试模式
# with torch.no_grad():
runner.fit(experiment, datamodule=data)
  #  pass
# print('before_fc',a)
# print('after_encoder',list(model.parameters()))
# print('after_fc',list(model.parameters())[41])
print("*")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
        
# torch.save(model, '/content/drive/MyDrive/TUM/PyTorch-VAE/model_2stage_100mResolution.pth')

# Path = '/content/drive/MyDrive/TUM/PyTorch-VAE/model_2_stage_200bs.pth'
# torch.save({
#             # 'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             # 'optimizer_state_dict': optimizer.state_dict(),
#             # 'loss': loss,
#             # ...
#             }, Path)

# print('done!')


import os
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import sys

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self, transform):
      # LiDAR data with (300, 13, 511, 511)
        data_path="/content/drive/MyDrive/sub_sample/dataset/Who_imgs_train_data_500_1000_13bands.npy" #"/content/drive/MyDrive/sub_sample/imgs_train_data_0_1500_13bands.npy"

        lst_path = "/content/drive/MyDrive/sub_sample/dataset/Who_imgs_train_data_500_1000_LST_bands.npy"
      # LST data with (300, 1, 511, 511)
        imgs_test = np.load(data_path)

        imgs_test = imgs_test[0:50,:,:,:]
        imgs_lst = np.load(lst_path)
        imgs_lst = imgs_lst[0:50,:,:,:]
        print(imgs_test.shape,imgs_lst.shape)
        print(imgs_test.max(),imgs_test.min(),imgs_lst.max(),imgs_lst.min())

        self.transform = torchvision.transforms.Resize((512,512))

        imgg = imgs_test.astype('float32')
        # if np.ptp(imgg,axis = (2,3), keepdims=True).any()==0:
        imgs_test = (imgg - np.min(imgg,axis =(2,3),keepdims=True)) / (0.00001+np.ptp(imgg,axis = (2,3), keepdims=True))#axis=(0,2,3),
        imgs = torch.from_numpy(imgs_test)
        imgs_test = self.transform(imgs)
        imgs_test = imgs_test.numpy()

        imgg_lst = imgs_lst.astype('float32')
        # if np.ptp(imgg,axis = (2,3), keepdims=True).any()==0:
        imgs_test_lst = imgg_lst # (imgg_lst - np.min(imgg_lst,axis =(2,3),keepdims=True)) / (0.00001+np.ptp(imgg_lst,axis = (2,3), keepdims=True))#axis=(0,2,3),
        imgs = torch.from_numpy(imgs_test_lst)        
        imgs_test_lst = self.transform(imgs)
        imgs_test_lst = imgs_test_lst.numpy()
        print(imgs_test_lst.shape)
        # seperate into 64 * 64 patches
        box_list = []
        lst_list = []
        lidar_list = []
        item_width = 64
        count=0
        # for j in range(0,8):
        #   for i in range(0,8):
        #     count+=1
        #     box=(i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width)
        #     box_list.append(box)

        lidar_list = np.concatenate([imgs_test[:,:,i*item_width:(i+1)*item_width,j*item_width:(j+1)*item_width ] for  i in range(8) for j in range(8)],axis = 0)
        lst_list = np.concatenate([imgs_test_lst[:,:,i*item_width:(i+1)*item_width,j*item_width:(j+1)*item_width ] for i in range(8) for j in range(8)],axis = 0)
        # for i in range(64):
        # image_list[i]=[imgs_test[box] for box in box_list]
        # box_list = []
        # for j in range(0,8):
        #   for i in range(0,8):
        #     count+=1
        #     box=(i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width)
        #     box_list.append(box)
        # lst_list = [imgs_test_lst.append(imgs_test_lst[box]) for box in box_list]
        # a = c
        # a_final = np.ndarray(shape = (300,13,64,64),dtype = np.float64)
        # for i in range(300):
        #   a_final[i] = self.transform(imgs_test[i,0:13,448:-1,448:-1])
        # aa = np.concatenate((a,a_final),axis=0)
        # d = imgs_test_lst[:,0,0:64,0:64]
        # for m in range(0,6):
        lidar_list = np.array(lidar_list)
        lst_list = np.array(lst_list)
        #   # img1  = imgs_test[:,0:13,n*64:(n+1)*64,n*64:(n+1)*64]
        #   img2_lst  = imgs_test_lst[:,0,(m+1)*64:(m+2)*64,(m+1)*64:(m+2)*64]
        #   # c = img1
        #   img1_lst = np.concatenate((c, img2_lst),axis = 0) #imgs_test[0:300,2:5,50:114,50:114]
        #   d = img1_lst
        #   # g = np.concatenate((q, img2),axis = 0)
        # b = d
        # b_final = self.transform(imgs_test_lst[:,0,448:-1,448:-1])
        image_list = np.concatenate((lidar_list,lst_list),axis=1)
        print(image_list.shape)
        
        # print('***********************',image_list[:,0:13,:,:],'------------------------')
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',image_list[:,14,:,:],'&&&&&&&&&&&&&&&&&&&&&&')

        # imgs_test =   np.concatenate((b,a),axis = 1)
      

        self.training = image_list 

    def __getitem__(self, index):

        img = self.training[index, :, :, :]  # 读取每一个npy的数据

        img = torch.from_numpy(img)
        
        self.transform1 = torchvision.transforms.Resize((128,128))
        imgs = self.transform1(img)
        return imgs
        
    def __len__(self):
        # print('the shape of the total dataset',len(self.training))
        return len(self.training)


    # def cut_image(image):
    #     width = 512
    #     length = 512
    #     item_width=int(width/8)
    #     box_list=[]
    #     count=0
    #     for j in range(0,8):
    #       for i in range(0,8):
    #         count+=1
    #         box=(i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width)
    #         box_list.append(box)
    #     print(count)
    #     image_list=[image.crop(box) for box in box_list]
    #     return image_list
 

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int =1,
        val_batch_size: int = 1,
        # patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        # self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================

        # whole_data = load_train_data(data_path)

        # whole_dataset = torch.utils.data.TensorDataset(torch.Tensor(whole_data))
        # train_dataset = torch.vstack(train_dataset)

        train_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148),
                                              # lambda x: Image.fromarray(np.uint8(x),multichannel=True).convert('RGB'),
                                              
                                              # transforms.Resize((256,256)),
                                              transforms.ToTensor(),
                                              # transforms.ToPILImage(),
                                              
                                             ])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size)])
        #                                     # transforms.ToTensor(),])
        
        self.whole_dataset = MyDataset(
            transform=train_transforms
        )
         # Replace CelebA with your dataset
        # self.val_dataset = MyDataset(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        lengths = [2560,640]  # 
        # lengths = [300,20]
        # lengths = [660,300]
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.whole_dataset, lengths, generator=torch.Generator().manual_seed(5))
        # generator=torch.Generator().manual_seed(5))
        #

        # print(f"Label: {label}")
                
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=512,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=512,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            # pin_memory=self.pin_memory,
        )
    # def predict_dataloader(self) -> DataLoader:

    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.train_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size= 512,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.train_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )

    

        # train_generator = DataLoader()
        # test_generator = DataLoader()
#         print('The whole data set:', len(self.whole_dataset))
#         print('Test data set:', len(self.train_set))
#         print('Valid data set:', len(self.val_set))
# #       ===============================================================



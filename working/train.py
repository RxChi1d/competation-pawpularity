#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[ ]:


import os
import gc
import math
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.append("../input/pythonbox")
from box import Box

# Torch
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset 

# Model
# 根據需求更改
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')  
from timm import create_model
from torchvision.models import efficientnet_v2_l

# Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# ## Config

# In[ ]:


config = {'exp_name':'exp_20230521-2',
          'commit': "swin_large_patch4_window7_224_in22k(f10_e50_bs32)",
          'root': '../input/petfinder-pawpularity-score/',  # Data root
          'seed': 2023,
          'n_splits': 10,
          'n_epochs': 50,
          'early_stop': 10,
          'image_size': 224,
          'lr': 1e-4,  # 如果有使用lr_find這個值會自動更改
          'lr_find': {'max_lr': 1e-4,
                      'min_lr': 1e-6,
                      'num_training': 100},
          'model':{
              'package': 'timm',  # timm or torchvision
              'name': 'swin_large_patch4_window7_224_in22k',
              'output_dim': 1,
              'pretrain': True,
          },
          'save_dir': 'swin_large_patch4_window7_224_in22k(f10_e50_bs32)',  # 儲存權重與log的資料夾
          'train_loader': {
              'batch_size': 32,
              'shuffle': True,
              'num_workers': os.cpu_count(),
              'pin_memory': True,
              'drop_last': False
          },
          'val_loader': {
              'batch_size': 32,
              'shuffle': False,
              'num_workers': os.cpu_count(),
              'pin_memory': True,
              'drop_last': False
          },
          'loss': 'RMSE',
}

config = Box(config)


# ## Fix Seed

# In[ ]:


seed_everything(config.seed)


# ## Tools

# In[ ]:


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


def RMSE(predict,target):
    return torch.sqrt(nn.MSELoss()(predict.float(), target.float()))

def MSE(predict,target):
    return nn.MSELoss()(predict.float(), target.float())


# ## Dataset

# In[ ]:


class PetfinderDataset(Dataset):
    """Dataset
    Args:
        df: the dataframe from csv, and the "Id" column needs to be the path of Image
    """
    def __init__(self, df, transform=None, image_size=224):
        
        self._X = df["Id"].values
        self._y = None
        self.transform = transform
        
        # 判斷有沒有分數
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
            
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self.transform(image)
        
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image
    
    
class PetfinderDataModule(LightningDataModule):
    """PetfinderDataModule
    Args:
        cfg (dict): config
        fold (int): idx of fold
        stage (srt): train or test
    """
    def __init__(self, cfg, fold, stage="train"):
        super().__init__()
        df = pd.read_csv(os.path.join(config.root, stage+'.csv'))
        df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, stage, x + ".jpg")) # 將ID改成圖片路徑
        df["Pawpularity"] = df["Pawpularity"].astype(float).apply(lambda x: x / 100.) # 將Pawpularity軟換到[0, 1]
        # df = transform_get_image_shape_describe(df)  # 應該是用來增加額外的meta data
        
        self._df = df
        self._cfg = cfg
        self.fold = fold
        self.train_transform, self.val_transform = self.get_transforms()

    def setup(self, stage: str):
        if (not os.path.isfile('dataset_temp.pickle')) or (self.fold == 0):
            # 創建kfold的dataframe index
            kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
            all_splits = [(x.tolist(), y.tolist()) for x, y in kf.split(self._df)]

            # 將list of indexs存成緩存檔案
            with open('dataset_temp.pickle', 'wb') as f:
                pickle.dump(all_splits, f)
                
        # 讀取list of indexs，並取出本次fold資料分配
        with open('dataset_temp.pickle', 'rb') as f:
            all_splits = pickle.load(f)

        train_indexes, val_indexes = all_splits[self.fold]
        self.train_df, self.valid_df = self._df.iloc[train_indexes], self._df.iloc[val_indexes]

        # df to dataset
        self.train_data = PetfinderDataset(self.train_df, self.train_transform, self._cfg.image_size)
        self.val_data = PetfinderDataset(self.valid_df, self.val_transform, self._cfg.image_size)
                
    def get_transforms(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
        IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
        
        train_transform = T.Compose([T.Resize(self._cfg.image_size),
                                     T.CenterCrop([self._cfg.image_size, self._cfg.image_size]),
                                     T.RandomHorizontalFlip(),
                                     T.RandomVerticalFlip(),
                                     T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                                     T.ConvertImageDtype(torch.float),
                                     T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        
        val_transform = T.Compose([T.Resize(self._cfg.image_size),
                                   T.CenterCrop([self._cfg.image_size, self._cfg.image_size]),
                                   T.ConvertImageDtype(torch.float),
                                   T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        
        return train_transform, val_transform
    
    def train_dataloader(self):
        return DataLoader(self.train_data, **self._cfg.train_loader)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, **self._cfg.val_loader)


# ## Model

# In[ ]:


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)  # 儲存超參數
        
        if 'nn' in self.hparams.loss:
            self._criterion = eval(self.hparams.loss)()
        else:
            self._criterion = eval(self.hparams.loss)
        self.metrics = RMSE
        
        self.validation_step_outputs = {'val/logits': [],
                                        'val/pred': [],
                                        'val/labels': []}  # 用來計算epoch的val/loss, val/rmse
        
        self.__build_model()
        
    def __build_model(self):
        if self.hparams.model.package == 'timm':
            self.backbone = create_model(self.hparams.model.name,
                                         pretrained=self.hparams.model.pretrain, 
                                         num_classes=0, 
                                         in_chans=3)
            num_features = self.backbone.num_features
        
        elif self.hparams.model.package == 'torchvision':
            weights = 'DEFAULT' if self.hparams.model.pretrain else None
            self.backbone = eval(self.hparams.model.name)(weights=weights)
            num_features = 1000
            
        self.fc = nn.Sequential(nn.Dropout(0.5), 
                                nn.Linear(num_features, 
                                          self.hparams.model.output_dim))

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Mixup (50%的機率)
        if torch.rand(1)[0] < 0.5:
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self(mix_images).squeeze().sigmoid()
            loss = self._criterion(logits, target_a) * lam + (1 - lam) * self._criterion(logits, target_b)
        else:
            logits = self(images).squeeze().sigmoid()
            loss = self._criterion(logits, labels)
            
        self.log("train/loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        logits = self(images).squeeze().sigmoid()
        loss = self._criterion(logits, labels)
        
        self.validation_step_outputs['val/logits'].append(logits)
        self.validation_step_outputs['val/labels'].append(labels)
        
        return {'val/loss': loss}
    
    def on_validation_epoch_end(self):
        logits = torch.cat(self.validation_step_outputs['val/logits'], dim=0)
        labels = torch.cat(self.validation_step_outputs['val/labels'], dim=0)
        pred = logits
        loss = self._criterion(logits, labels)
        metric = self.metrics(pred, labels) * 100.
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/metric', metric, prog_bar=True)
        
        self.validation_step_outputs['val/logits'].clear()
        self.validation_step_outputs['val/pred'].clear()
        self.validation_step_outputs['val/labels'].clear()

    def predict_step(self, batch, batch_idx):
        images = batch
        pred = self(images).squeeze().sigmoid()  # return後會自動轉成numpy
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


# ## Train

# ### Callbacks

# In[ ]:


# 解決inf/nan無法於tqdm中顯示的問題，將其轉換為None
def convert_inf(x):
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


# 使TQDMProgressBar顯示fold的進度
class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Fold {self.fold}, Epoch {trainer.current_epoch}")
        

# 將常用的參數設為預設（只是為了簡化kfold loop中的code）
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            save_top_k=1,
            monitor="val/metric",
            mode="min",
            filename="metric_{val/metric:.3f}-epoch_{epoch:02d}",
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False,
            **kwargs
        )
        
        
class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            monitor="val/metric",
            patience=config.early_stop,
            mode="min",
            # verbose=False,
            **kwargs
        )


# ### Train Loop

# In[ ]:


warnings.filterwarnings("ignore")  # 關閉 Warning
torch.set_float32_matmul_precision('high')  # 設置高精度(根據顯卡調整)

for fold in range(config.n_splits):  
    datamodule = PetfinderDataModule(config, fold=fold, stage="train")  # 載入data

    # Auto find lr
    if (config.lr_find is not None) and (config.lr_find):
        model = Model(config)  # 載入model
        
        trainer = pl.Trainer(max_epochs=config.n_epochs, 
                        precision='16-mixed', # AMP 自動混合精度
                        benchmark=True,  # 加速運算
                        logger=False
                        )

        tuner = pl.tuner.tuning.Tuner(trainer)

        tuner.lr_find(model, datamodule=datamodule, **config.lr_find)
        
        new_lr = model.hparams.lr
        config.lr = new_lr
        print(f"lr: {new_lr}")
        
    model = Model(config)  # 載入model
    
    # 宣告功能
    bar = CustomTQDMProgressBar(fold)
    tb_logger = TensorBoardLogger(save_dir=config.save_dir, name=f'fold_{fold}')
    checkpoint_callback = CustomModelCheckpoint(dirpath=tb_logger.log_dir)
    early_stop_callback = CustomEarlyStopping()
    
    trainer = pl.Trainer(max_epochs=config.n_epochs, 
                        precision='16-mixed', # AMP 自動混合精度
                        benchmark=True,  # 加速運算
                        callbacks=[bar, checkpoint_callback, early_stop_callback],  # 設定callback，可以指定多個callbacks
                        logger=tb_logger,
                        enable_model_summary=False, # 不顯示模型架構的資訊
                        accelerator='gpu',
                        devices=-1
                        # num_sanity_val_steps=1,  # 在正式訓練前先跑一次val_loop做測試，確保val_loop是正確的
                        )

    trainer.fit(model, datamodule=datamodule)
    
    # 刪除不必要的變數與cache，避免爆內存
    del datamodule, model, bar, tb_logger, checkpoint_callback,         early_stop_callback, trainer

    torch.cuda.empty_cache()
    gc.collect()


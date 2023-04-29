import cv2
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch.nn.functional import mse_loss
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor
from tqdm import tqdm


class args():
    def __init__(self):
        self.DEVICE = "cuda"
        self.EPOCHS = 20
        self.GAMMA = 0.1
        self.KFOLD = 3
        self.LEARNING_RATE = 1e-4
        self.SEED = 42
        self.STEPS = 10
        self.TRAINING_BATCH_SIZE = 16
        self.TESTING_BATCH_SIZE = 1


def build_environment(cfg):
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Paw(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        self.path_train = "../input/petfinder-pawpularity-score/train/"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # image features
        image_id = self.dataframe["Id"][idx]
        image_path = os.path.join(self.path_train, image_id + ".jpg")
        image = cv2.imread(image_path)
        image = self.transform(image)

        # other features
        focus = self.dataframe["Subject Focus"][idx]
        eyes = self.dataframe["Eyes"][idx]
        face = self.dataframe["Face"][idx]
        near = self.dataframe["Near"][idx]
        action = self.dataframe["Action"][idx]
        accessory = self.dataframe["Accessory"][idx]
        group = self.dataframe["Group"][idx]
        collage = self.dataframe["Collage"][idx]
        human = self.dataframe["Human"][idx]
        occlusion = self.dataframe["Occlusion"][idx]
        info = self.dataframe["Info"][idx]
        blur = self.dataframe["Blur"][idx]
        all_features = [focus, eyes, face, near, action,
                        accessory, group, collage, human, occlusion, info, blur]
        features = np.array(all_features).astype(np.float32)

        # label
        pawpularity = self.dataframe["Pawpularity"][idx]
        label = np.array(pawpularity).astype(np.float32)

        return [image, features, label]


class Data():
    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self):
        self.training_transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Resize((224, 224))
        ])
        self.testing_transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Resize((224, 224))
        ])

    def build_data(self):
        dataframe = pd.read_csv(
            "../input/petfinder-pawpularity-score/train.csv")

        self.training_data = Paw(dataframe, transform=self.training_transforms)
        self.testing_data = Paw(dataframe, transform=self.testing_transforms)

        # train_df, val_df = train_test_split(
        #     dataframe, test_size=0.2, random_state=self.cfg.SEED)
        # train_df = train_df.reset_index(drop=True)
        # val_df = val_df.reset_index(drop=True)

        # self.training_data = Paw(train_df, transform=self.training_transforms)
        # self.testing_data = Paw(val_df, transform=self.testing_transforms)

    def build_dataloader(self):
        self.training_dataloader = None
        self.testing_dataloader = None
        # self.training_dataloader = DataLoader(
        #     self.training_data, batch_size=self.cfg.TRAINING_BATCH_SIZE, shuffle=True)
        # self.testing_dataloader = DataLoader(
        #     self.testing_data, batch_size=self.cfg.TESTING_BATCH_SIZE, shuffle=False)


class PawNet(nn.Module):
    def __init__(self):
        super(PawNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fc1 = nn.Linear(1000 + 12, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, image, features):
        image = self.resnet(image)                      # (batch_size, 1000)
        image = image.view(image.size(0), -1)           # (batch_size, 1000)
        concat = torch.cat((image, features), dim=1)    # (batch_size, 1012)
        concat = self.fc1(concat)                       # (batch_size, 512)
        concat = nn.functional.relu(concat)             # (batch_size, 512)
        concat = self.fc2(concat)                       # (batch_size, 1)
        return concat


def rmse(input, target):
    return torch.sqrt(mse_loss((input.flatten()), target))


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg

    def build_model(self):
        self.net = PawNet()
        self.net.to(self.cfg.DEVICE)

    def build_tools(self):
        self.criterion = rmse
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.cfg.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.cfg.STEPS, gamma=self.cfg.GAMMA)


def train(epoch, cfg, data, trainer):
    trainer.net.train()

    loss_training = []

    with tqdm(total=len(data.training_dataloader), desc=f'Training Epoch {epoch+1}/{cfg.EPOCHS}', unit='BATCH') as pbar_training:
        for batch, (image, features, label) in enumerate(data.training_dataloader):
            image, features, label = image.to(cfg.DEVICE), features.to(
                cfg.DEVICE), label.to(cfg.DEVICE)

            pred = trainer.net(image, features)
            loss = trainer.criterion(pred, label)
            loss_training.append(loss.item())

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            pbar_training.update(1)
            pbar_training.set_postfix(**{'loss': loss.item()})

    return {"loss": np.mean(loss_training)}


def validate(epoch, cfg, data, trainer):
    trainer.net.eval()

    loss_validation = []

    with tqdm(total=len(data.testing_dataloader), desc=f'Validation Epoch {epoch+1}/{cfg.EPOCHS}', unit='BATCH') as pbar_testing:
        for batch, (image, features, label) in enumerate(data.testing_dataloader):
            image, features, label = image.to(cfg.DEVICE), features.to(
                cfg.DEVICE), label.to(cfg.DEVICE)

            pred = trainer.net(image, features)
            loss = trainer.criterion(pred, label)
            loss_validation.append(loss.item())

            pbar_testing.update(1)
            pbar_testing.set_postfix(**{'loss': loss.item()})

    return {"loss": np.mean(loss_validation)}


def main(cfg):
    kfold = KFold(n_splits=cfg.KFOLD, shuffle=True)

    data = Data(cfg)
    data.build_transforms()
    data.build_data()
    data.build_dataloader()

    val_loss_kfold = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data.training_data)):

        print("Fold {}:".format(fold + 1))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        data.training_dataloader = DataLoader(
            data.training_data, batch_size=cfg.TRAINING_BATCH_SIZE, sampler=train_subsampler)
        data.testing_dataloader = DataLoader(
            data.testing_data, batch_size=cfg.TESTING_BATCH_SIZE, sampler=val_subsampler)

        trainer = Trainer(cfg)
        trainer.build_model()
        trainer.build_tools()

        epochs = []
        trn_loss_epochs = []
        val_loss_epochs = []

        for epoch in range(cfg.EPOCHS):
            training_results = train(epoch, cfg, data, trainer)
            validtion_results = validate(epoch, cfg, data, trainer)

            trainer.scheduler.step()

            print("Training loss: {}".format(training_results["loss"]))
            print("Validation loss: {}".format(validtion_results["loss"]))
            epochs.append(epoch)
            trn_loss_epochs.append(training_results["loss"])
            val_loss_epochs.append(validtion_results["loss"])

        val_loss_kfold.append(np.mean(val_loss_epochs))
        print("Best validation loss: {}".format(min(val_loss_epochs)))
        print("Average validation loss: {}".format(np.mean(val_loss_epochs)))
        print("---------------------------------------------------------------")

        # fig_loss = plt.figure(figsize=(10, 10))
        # ax_loss = fig_loss.add_subplot(1, 1, 1)
        # ax_loss.plot(epochs, trn_loss_epochs,
        #              label="Training Loss", color='tab:blue')
        # ax_loss.plot(epochs, val_loss_epochs,
        #              label="Validation Loss", color='tab:orange')
        # ax_loss.set_title('loss')
        # ax_loss.set_xlabel('epochs')
        # ax_loss.set_ylabel('loss')
        # ax_loss.legend()
        # plt.show()

    print("Average validation loss for all folds: {}".format(
        np.mean(val_loss_kfold)))
    print("Finished !\n")


if __name__ == '__main__':
    arguments = args()
    build_environment(arguments)
    main(arguments)

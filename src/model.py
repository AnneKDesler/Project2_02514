from typing import Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from src.loss import binary_focal_loss_with_logits

def bce_loss(pred, target):	
    return torch.mean(F.binary_cross_entropy_with_logits(pred, target))

def focal_loss(pred, target):	
    return torch.mean(binary_focal_loss_with_logits(pred, target))

class Model(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 1,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 1,
        batch_normalization: Optional[bool] = False,
        optimizer: Optional[str] = None,
        target_mask_supplied: Optional[bool]=False,
        loss = None,
        *args,
        **kwargs
    ) -> None:
        super(Model, self).__init__(*args, **kwargs)
        
        self.target_mask_supplied = target_mask_supplied

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.downsample0 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.enc_conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                       nn.ReLU())
        self.downsample1 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        self.enc_conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.downsample2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.enc_conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.downsample3 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                                             nn.ReLU())

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        
        self.dec_conv0 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        
        self.dec_conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        
        self.dec_conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                       nn.ReLU())
        
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        self.dec_conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, num_classes, 1, stride=1))
        self.lr = lr
        self.batch_size = batch_size
        if loss == "focal":
            self.loss = focal_loss
        else:
            if num_classes == 1:
                # self.loss = BinaryFocalLossWithLogits(alpha=0.25, reduction="mean")
                self.loss = bce_loss
            else:
                self.loss = torch.nn.CrossEntropyLoss()

        if optimizer is None or optimizer == "Adam":
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x: List[str]) -> List[str]:
        """
        https://huggingface.co/docs/transformers/model_doc/t5#inference
        """

        # encoder
        e0 = self.enc_conv0(x)
        e0_down = self.downsample0(e0)
        e1 = self.enc_conv1(e0_down)
        e1_down = self.downsample1(e1)
        e2 = self.enc_conv2(e1_down)
        e2_down = self.downsample2(e2)
        e3 = self.enc_conv3(e2_down)
        e3_down = self.downsample3(e3)

        # bottleneck
        b = self.bottleneck_conv(e3_down)
        # decoder
        b = self.upsample0(b)
        b = torch.cat((b, e3), dim=1)  # skip-connection
        d0 = self.dec_conv0(b)

        d0 = self.upsample1(d0)

        d0 = torch.cat((d0, e2), dim=1)  # skip-connection
        d1 = self.dec_conv1(d0)

        d1 = self.upsample2(d1)

        d1 = torch.cat((d1, e1), dim=1)  # skip-connection
        d2 = self.dec_conv2(d1)

        d2 = self.upsample3(d2)

        d2 = torch.cat((d2, e0), dim=1)  # skip-connection
        d3 = self.dec_conv3(d2)  # no activation
        
        return d3

    def _inference_training(
        self, batch, batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        From https://huggingface.co/docs/transformers/model_doc/t5#training
        """
        if self.target_mask_supplied:
            data, target, mask = batch
        else:
            data, target = batch
        output = self(data)
        if self.target_mask_supplied:
            output *= mask[:,None,:,:]

        output, target = output, target
        #print(output.shape, target.shape)
        output = output[:,0,:,:]
        target = target.to(torch.float32)
        dice, iou, accuracy, sensitivity, specificity = self.metrics(output, target)

        #out_img = wandb.Image(
        #    output[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="Prediction"
        #)
        #out_target = wandb.Image(
        #    target[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="target"
        #)
        #self.logger.experiment.log({"prediction": [out_img, out_target]}) #, step = self.logger.experiment.current_trainer_global_step

        return self.loss(output, target), accuracy, specificity, iou, dice, sensitivity

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train accuracy", accuracy, batch_size=self.batch_size)
        self.log("train specificity", specificity, batch_size=self.batch_size)
        self.log("train iou", iou, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("val specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("val iou", iou, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("test specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("test iou", iou, batch_size=self.batch_size, sync_dist=True)
        self.log("test dice", dice, batch_size=self.batch_size, sync_dist=True)
        self.log("test sensitivity", sensitivity, batch_size=self.batch_size, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def metrics(self, preds, target):
        # Dice
        X = target.view(-1)
        Y = torch.sigmoid(preds.view(-1)) > 0.5

        Y = Y*1.0
        dice = 2*torch.mean(torch.mul(X,Y))/torch.mean(X+Y)

        # Intersection over Union
        IoU = torch.mean(torch.mul(X,Y))/(torch.mean(X+Y)-torch.mean(torch.mul(X,Y)))

        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(X, Y).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)		
        
        # Sensitivity
        sensitivity = tp/(tp+fn)

        # Specificity
        specificity = tn/(tn+fp)

        return dice, IoU, accuracy, sensitivity, specificity













class DilatedNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 1,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 1,
        optimizer: Optional[str] = None,
        target_mask_supplied: Optional[bool]=False,
        loss = None,
        *args,
        **kwargs
    ) -> None:
        super(DilatedNet, self).__init__(*args, **kwargs)

        self.target_mask_supplied = target_mask_supplied

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, dilation=1)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, dilation=2)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, dilation=4)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, dilation=8)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.dec_conv0 = nn.ConvTranspose2d(64, 64, 3, dilation=8)
        self.dec_conv1 = nn.ConvTranspose2d(64, 64, 3, dilation=4)
        self.dec_conv2 = nn.ConvTranspose2d(64, 64, 3, dilation=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 1, 3, dilation=1)

        self.lr = lr
        self.batch_size = batch_size
        if loss == "focal":
            self.loss = focal_loss
        else:
            if num_classes == 1:
                # self.loss = BinaryFocalLossWithLogits(alpha=0.25, reduction="mean")
                self.loss = bce_loss
            else:
                self.loss = torch.nn.CrossEntropyLoss()

        if optimizer is None or optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(b))
        d1 = F.relu(self.dec_conv1(d0))
        d2 = F.relu(self.dec_conv2(d1))
        d3 = self.dec_conv3(d2)
        return d3

    def _inference_training(
        self, batch, batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        From https://huggingface.co/docs/transformers/model_doc/t5#training
        """
        if self.target_mask_supplied:
            data, target, mask = batch
        else:
            data, target = batch
        output = self(data)
        if self.target_mask_supplied:
            output *= mask[:,None,:,:]

        output, target = output, target
        #print(output.shape, target.shape)
        output = output[:,0,:,:]
        target = target.to(torch.float32)
        dice, iou, accuracy, sensitivity, specificity = self.metrics(output, target)

        #out_img = wandb.Image(
        #    output[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="Prediction"
        #)
        #out_target = wandb.Image(
        #    target[0,...].cpu().detach().numpy().squeeze(), 
        #    caption="target"
        #)
        #self.logger.experiment.log({"prediction": [out_img, out_target]}) #, step = self.logger.experiment.current_trainer_global_step

        return self.loss(output, target), accuracy, specificity, iou, dice, sensitivity

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train accuracy", accuracy, batch_size=self.batch_size)
        self.log("train specificity", specificity, batch_size=self.batch_size)
        self.log("train iou", iou, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("val specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("val iou", iou, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy, specificity, iou, dice, sensitivity = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        self.log("test iou", iou, batch_size=self.batch_size, sync_dist=True)
        self.log("test specificity", specificity, batch_size=self.batch_size, sync_dist=True)
        self.log("test dice", dice, batch_size=self.batch_size, sync_dist=True)
        self.log("test sensitivity", sensitivity, batch_size=self.batch_size, sync_dist=True)


        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def metrics(self, preds, target):
        # Dice
        X = target.view(-1)
        Y = torch.sigmoid(preds.view(-1)) > 0.5

        Y = Y*1.0
        dice = 2*torch.mean(torch.mul(X,Y))/torch.mean(X+Y)

        # Intersection over Union
        IoU = torch.mean(torch.mul(X,Y))/(torch.mean(X+Y)-torch.mean(torch.mul(X,Y)))

        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(X, Y).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)		
        
        # Sensitivity
        sensitivity = tp/(tp+fn)

        # Specificity
        specificity = tn/(tn+fp)

        return dice, IoU, accuracy, sensitivity, specificity
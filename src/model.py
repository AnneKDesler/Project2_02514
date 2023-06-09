from typing import Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F



class Model(pl.LightningModule):
    def __init__(
        self,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 1,
        batch_normalization: Optional[bool] = False,
        optimizer: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super(Model, self).__init__(*args, **kwargs)
        print("hello")
        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.downsample0 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.enc_conv1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                       nn.Conv2d(128, 128, 3, stride=1, padding=1))
        self.downsample1 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        
        self.enc_conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                       nn.Conv2d(256, 256, 3, stride=1, padding=1))
        self.downsample2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.enc_conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                       nn.Conv2d(512, 512, 3, stride=1, padding=1))
        self.downsample3 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                             nn.Conv2d(1024, 1024, 3, stride=1, padding=1))

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        
        self.dec_conv0 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1),
                                       nn.Conv2d(512, 512, 3, stride=1, padding=1))
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        
        self.dec_conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1),
                                       nn.Conv2d(256, 256, 3, stride=1, padding=1))
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        
        self.dec_conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1),
                                       nn.Conv2d(128, 128, 3, stride=1, padding=1))
        
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        self.dec_conv3 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                       nn.Conv2d(64, 1, 3, stride=1, padding=1))
        self.lr = lr
        self.batch_size = batch_size
        self.loss = torch.nn.BCELoss()
        if optimizer is None or optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x: List[str]) -> List[str]:
        """
        https://huggingface.co/docs/transformers/model_doc/t5#inference
        """
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e0_down = self.downsample0(e0)
        e1 = F.relu(self.enc_conv1(e0_down))
        e1_down = self.downsample1(e1)
        e2 = F.relu(self.enc_conv2(e1_down))
        e2_down = self.downsample2(e2)
        e3 = F.relu(self.enc_conv3(e2_down))
        e3_down = self.downsample3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3_down))
        # decoder
        b = self.upsample0(b)

        b = torch.cat((b, e3), dim=1)  # skip-connection
        d0 = F.relu(self.dec_conv0(b))

        d0 = self.upsample1(d0)

        d0 = torch.cat((d0, e2), dim=1)  # skip-connection
        d1 = F.relu(self.dec_conv1(d0))

        d1 = self.upsample2(d1)

        d1 = torch.cat((d1, e1), dim=1)  # skip-connection
        d2 = F.relu(self.dec_conv2(d1))

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

        data, target = batch
        output = self(data)
        output, target = output.flatten(), target.flatten()
        preds = torch.round(output)
        accuracy = self.accuracy(preds, target)
        target = target.to(torch.float32)

        return self.loss(output, target), accuracy

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train accuracy", accuracy, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def accuracy(self, preds, target):
        return (preds == target).sum() / len(target)
    
    
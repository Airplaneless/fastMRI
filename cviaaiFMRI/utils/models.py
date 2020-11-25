import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from fastmri.pl_modules import FastMriDataModule, MriModule
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform


class ENet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, decoder_channels=128, dropout=0.1):
        super().__init__()

        self.net = nn.ModuleList([
            DownSampler(in_channels, decoder_channels//8),
            Bottleneck(decoder_channels//8, decoder_channels//2, dropout/10, downsample=True),

            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout/10),

            Bottleneck(decoder_channels//2, decoder_channels, dropout, downsample=True),

            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=2),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=4),
            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=8),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=16),

            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=2),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=4),
            Bottleneck(decoder_channels, decoder_channels, dropout),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=8),
            Bottleneck(decoder_channels, decoder_channels, dropout, asymmetric_ksize=5),
            Bottleneck(decoder_channels, decoder_channels, dropout, dilation=16),

            UpSampler(decoder_channels, decoder_channels//2),

            Bottleneck(decoder_channels//2, decoder_channels//2, dropout),
            Bottleneck(decoder_channels//2, decoder_channels//2, dropout),

            UpSampler(decoder_channels//2, decoder_channels//8),

            Bottleneck(decoder_channels//8, decoder_channels//8, dropout),

            nn.ConvTranspose2d(decoder_channels//8, out_channels, (2, 2), (2, 2))])

    def forward(self, x):
        max_indices_stack = []

        for module in self.net:
            if isinstance(module, UpSampler):
                x = module(x, max_indices_stack.pop())
            else:
                x = module(x)

            if type(x) is tuple:
                x, max_indices = x
                max_indices_stack.append(max_indices)

        return x


class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bt_channels = in_channels // 4

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1, 1), bias=False),
            nn.InstanceNorm2d(bt_channels),
            nn.ReLU(True),

            nn.ConvTranspose2d(bt_channels, bt_channels, (3, 3), 2, 1, 1),
            nn.InstanceNorm2d(bt_channels),
            nn.ReLU(True),

            nn.Conv2d(bt_channels, out_channels, (1, 1), bias=False),
            nn.InstanceNorm2d(out_channels))

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.InstanceNorm2d(out_channels))

    def forward(self, x, max_indices):
        x_skip_connection = self.skip_connection(x)
        x_skip_connection = F.max_unpool2d(x_skip_connection, max_indices, (2, 2))

        return F.relu(x_skip_connection + self.main_branch(x), inplace=True)


class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, (3, 3), 2, 1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = torch.cat((F.max_pool2d(x, (2, 2)), self.conv(x)), 1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
            self, in_channels, out_channels, dropout_prob=0.0, downsample=False,
            asymmetric_ksize=None, dilation=1, use_prelu=True):

        super().__init__()
        bt_channels = in_channels // 4
        self.downsample = downsample
        self.channels_to_pad = out_channels - in_channels

        input_stride = 2 if downsample else 1

        main_branch = [
            nn.Conv2d(in_channels, bt_channels, input_stride, input_stride, bias=False),
            nn.InstanceNorm2d(bt_channels),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True)
        ]

        if asymmetric_ksize is None:
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (3, 3), 1, dilation, dilation)
            ]
        else:
            assert type(asymmetric_ksize) is int
            ksize, padding = asymmetric_ksize, (asymmetric_ksize - 1) // 2
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (ksize, 1), 1, (padding, 0), bias=False),
                nn.Conv2d(bt_channels, bt_channels, (1, ksize), 1, (0, padding))
            ]

        main_branch += [
            nn.InstanceNorm2d(bt_channels),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True),
            nn.Conv2d(bt_channels, out_channels, (1, 1), bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.Dropout2d(dropout_prob)
        ]

        self.main_branch = nn.Sequential(*main_branch)
        self.output_activation = nn.PReLU(out_channels) if use_prelu else nn.ReLU(True)

    def forward(self, x):
        if self.downsample:
            x_skip_connection, max_indices = F.max_pool2d(x, (2, 2), return_indices=True)
        else:
            x_skip_connection = x

        if self.channels_to_pad > 0:
            x_skip_connection = F.pad(x_skip_connection, (0, 0, 0, 0, 0, self.channels_to_pad))

        x = self.output_activation(x_skip_connection + self.main_branch(x))

        if self.downsample:
            return x, max_indices
        else:
            return x


class UnetSMPModelPL(MriModule):

    def __init__(self, in_chans, out_chans, encoder_name, decoder_channels, lr, lr_step_size, lr_gamma, weight_decay, data_path, batch_size, mask_type, center_fractions, accelerations, decoder_attention_type, optim_eps):
        super().__init__()
        self.save_hyperparameters()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.encoder_name = encoder_name
        self.decoder_channels = decoder_channels
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.optim_eps = optim_eps
        self.net = self._make_net(in_chans, out_chans, encoder_name, decoder_channels, decoder_attention_type)
        mask = create_mask_for_mask_type(mask_type, center_fractions, accelerations)
        train_transform = UnetDataTransform('singlecoil', mask_func=mask, use_seed=False)
        val_transform = UnetDataTransform('singlecoil', mask_func=mask)
        test_transform = UnetDataTransform('singlecoil')
        self.data_module = FastMriDataModule(
            data_path=pathlib.Path(data_path),
            challenge='singlecoil',
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_split='test',
            test_path=None,
            sample_rate=1.0,
            batch_size=batch_size,
            num_workers=4,
            distributed_sampler=False
        )

    @staticmethod
    def _make_net(in_chans, out_chans, encoder_name, decoder_channels, decoder_attention_type):
        return smp.Unet(in_channels=in_chans, classes=out_chans, encoder_name=encoder_name, encoder_depth=len(decoder_channels), decoder_channels=decoder_channels, encoder_weights=None, decoder_attention_type=decoder_attention_type)

    def forward(self, image):
        return self.net(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _, _ = batch
        output = self(image)
        loss = F.l1_loss(output, target)
        self.log("loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value = batch
        output = self(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num, _ = batch
        output = self.forward(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.optim_eps)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)
        return [optim], [scheduler]


class EnetModelPL(MriModule):
    def __init__(self, in_chans, out_chans, dropout, decoder_channels, lr, lr_step_size, lr_gamma, weight_decay, data_path, batch_size, mask_type, center_fractions, accelerations, optim_eps):
        super().__init__()
        self.save_hyperparameters()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.decoder_channels = decoder_channels
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.optim_eps = optim_eps
        self.net = ENet(in_channels=in_chans, out_channels=out_chans, decoder_channels=decoder_channels, dropout=dropout)
        mask = create_mask_for_mask_type(mask_type, center_fractions, accelerations)
        train_transform = UnetDataTransform('singlecoil', mask_func=mask, use_seed=False)
        val_transform = UnetDataTransform('singlecoil', mask_func=mask)
        test_transform = UnetDataTransform('singlecoil')
        self.data_module = FastMriDataModule(
            data_path=pathlib.Path(data_path),
            challenge='singlecoil',
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            test_split='test',
            test_path=None,
            sample_rate=1.0,
            batch_size=batch_size,
            num_workers=4,
            distributed_sampler=False
        )

    def forward(self, image):
        return self.net(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _, _ = batch
        output = self(image)
        loss = F.l1_loss(output, target)
        self.log("loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value = batch
        output = self(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num, _ = batch
        output = self.forward(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.optim_eps)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)
        return [optim], [scheduler]
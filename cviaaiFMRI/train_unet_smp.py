import os
import pathlib
import argparse

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from cviaaiFMRI.utils.models import UnetSMPModelPL
pl.seed_everything(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mask_type',
        default='random',
        choices=('random', 'equispaced'),
        type=str,
        help='Mask type for k-space sampling'
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        '--encoder_name',
        default='resnet34',
        choices=list(smp.encoders.encoders.keys()),
        type=str,
        help='Encoder name for Unet blocks'
    )
    parser.add_argument(
        '--decoder_attention_type',
        default='None',
        choices=('None', 'scse'),
        type=str,
        help='attention module used in decoder of the model'
    )
    parser.add_argument(
        '--decoder_channels',
        nargs="+",
        default=[256, 128, 64, 32],
        type=int,
        help="Decoder channels in Unet blocks",
    )
    parser.add_argument(
        '--lr',
        default=3e-4,
        type=float,
        help='Learning rate',
    )
    parser.add_argument(
        '--lr_gamma',
        default=0.1,
        type=float,
        help='Learning rate gamma',
    )
    parser.add_argument(
        '--lr_step_size',
        default=40,
        type=int,
        help='Learning rate step size',
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float,
        help='Weight decay',
    )
    parser.add_argument(
        '--optim_eps',
        default=1e-8,
        type=float,
        help='Optimizer eps',
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='batch size',
    )
    parser.add_argument(
        '--data_path',
        default='D:\\source\\projects\\fastMRIdatasets\\',
        type=str,
        help='Path to fastMRI datasets',
    )

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        amp_backend='native',
        amp_level='O2',
        auto_lr_find=False,
        auto_scale_batch_size=False,
        auto_select_gpus=False,
        automatic_optimization=True,
        benchmark=False,
        check_val_every_n_epoch=1,
        deterministic=False,
        distributed_backend=None,
        default_root_dir='models/unet-unnamed',
        fast_dev_run=True,
        flush_logs_every_n_steps=100,
        max_epochs=10,
        gpus=1,
    )

    args = parser.parse_args()

    default_root_dir = pathlib.Path(args.default_root_dir)
    checkpoint_path = default_root_dir / 'checkpoints'

    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    ckpt_list = sorted(checkpoint_path.glob("*.ckpt"), key=os.path.getmtime)
    last_ckpt = str(ckpt_list[-1]) if ckpt_list else None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_top_k=True,
        verbose=True,
        monitor="validation_loss",
        mode="min",
        prefix=""
    )

    args.checkpoint_callback = checkpoint_callback
    args.resume_from_checkpoint = last_ckpt
    args.default_root_dir = default_root_dir

    model = UnetSMPModelPL(
        in_chans=1,
        out_chans=1,
        decoder_channels=args.decoder_channels,
        encoder_name=args.encoder_name,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        data_path=args.data_path,
        batch_size=args.batch_size,
        mask_type=args.mask_type,
        center_fractions=args.center_fractions,
        accelerations=args.accelerations,
        decoder_attention_type=None if args.decoder_attention_type is 'None' else args.decoder_attention_type,
        optim_eps=args.optim_eps
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=model.data_module)


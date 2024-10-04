"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.utils.logger import get_root_logger
from pointcept.utils.config import Config
import shutil
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import pointcept.utils.comm as comm
import wandb
import os
from pathlib import Path
from pointcept.utils import ForkedPdb
from wandb import sdk as wandb_sdk

from wandb.sdk.lib import telemetry


def send_object_to_processes(obj):
    if not comm.is_main_process():
        raise RuntimeError(
            f"send_object_to_processes called from rank={comm.get_rank()}!"
        )
    return_list = comm.all_gather(obj)
    for o in return_list:
        if o is not None:
            return o


def get_object_from_main():
    return_list = comm.all_gather(None)
    for o in return_list:
        if o is not None:
            return o


def train_on_config(cfg):
    # == Original code
    cfg = default_setup(cfg)

    # Original code
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def wandb_train(config_file, options, wandb_run=None):
    cfg = default_config_parser(config_file, options)
    cfg.passed_options = options

    # == WANDB
    if wandb_run is None and comm.is_main_process():
        exp_path = Path(cfg.save_path)
        exp_path.mkdir(exist_ok=True, parents=True)
        wandb_run = wandb.init(
            name=exp_path.name,
            project=exp_path.parent.name,
            config=cfg._cfg_dict,
            # magic=True,
        )
        if (cfg.weight) and (not Path(cfg.weight).is_file()):
            cfg.weight = wandb_run.use_model(cfg.weight)

        cfg = wandb_run.config

    if comm.is_main_process():
        cfg = Config(cfg.as_dict())
        send_object_to_processes(cfg)
    else:
        cfg = get_object_from_main()

    train_on_config(cfg)

    # Fine-tune
    if "FT_config" in cfg:

        # save on new folder
        options.update(
            weight=str(Path(cfg.save_path) / "model" / "model_last.pth"),
            save_path=get_new_save_path(cfg, "finetune"),
        )

        # Recurse new training
        wandb_train(cfg, options, wandb_run=wandb_run)


def train_on_cfg(cfg):
    # == Original code
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

    # Wandb model save
    if comm.is_main_process():
        filename = Path(cfg.save_path) / "model" / "model_last.pth"
        wandb.log_model(path=str(filename))

    del trainer


# def main_worker(cfg):
#     cfg = default_setup(cfg)
#     trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
#     trainer.train()

#     return trainer


def main(args):
    launch(
        wandb_train,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(
            args.config_file,
            args.options,
        ),
    )


def get_new_save_path(cfg, folder_name):
    true_savepath = Path(cfg.save_path) / folder_name
    true_savepath.mkdir(exist_ok=True, parents=True)
    (true_savepath / "model").mkdir(exist_ok=True, parents=True)
    fake_savepath = Path(cfg.save_path).parent / folder_name

    fake_savepath.unlink(missing_ok=True)
    os.symlink(
        str(true_savepath.absolute()),
        str(fake_savepath.absolute()),
        target_is_directory=True,
    )
    return fake_savepath


def change_save_path(cfg, folder_name):
    true_savepath = Path(cfg.save_path) / folder_name
    true_savepath.mkdir(exist_ok=True, parents=True)
    (true_savepath / "model").mkdir(exist_ok=True, parents=True)
    fake_savepath = Path(cfg.save_path).parent / folder_name

    fake_savepath.unlink(missing_ok=True)
    os.symlink(
        str(true_savepath.absolute()),
        str(fake_savepath.absolute()),
        target_is_directory=True,
    )


if __name__ == "__main__":
    # Pre-train
    args = default_argument_parser().parse_args()
    run = main(args)

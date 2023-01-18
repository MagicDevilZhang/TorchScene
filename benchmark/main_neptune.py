# modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import copy
import logging
import os
import time
import hydra
import torch
from typing import Dict
from omegaconf import DictConfig
from torch import nn, optim

from torch.utils.tensorboard import SummaryWriter

import neptune.new as neptune

from tools.dataloader import load_data
from tools.train import train
from utils.miscellaneous import mkdir, collect_env_info
from utils.model_zoo import initialize_model


@hydra.main(version_base=None, config_path="../conf", config_name="train_vit")
def main(cfgs: DictConfig):
    logger = logging.getLogger(cfgs.arch)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if cfgs.ml_track_framework == 'neptune':
        writer = neptune.init_run(
            # NOTE: add your own project name (no automatic)
            project="huaizheng-zhang/ml-track-bench",
            # NOTE: add your own api token to your environment variables
            tags="Basic script",
            source_files=["*.py"],
        )
        logger.info("neptune running")
    elif cfgs.ml_track_framework == 'tensorboard':
        writer = SummaryWriter(os.path.join(
        cfgs.tensorborad_log_dir, time.ctime().replace(' ', '_').replace(':', '-')))
        logger.info("tensorboard log dir: {}".format(writer.log_dir))
    
    # create model
    logger.info("getting model '{}' from torch hub".format(cfgs.arch))
    model, input_size = initialize_model(
        model_name=cfgs.arch,
        num_classes=cfgs.num_classes,
        feature_extract=cfgs.feature_extract,
        use_pretrained=cfgs.pretrained,
    )
    logger.info("model: '{}' is successfully loaded".format(
        model.__class__.__name__))
    logger.info("model structure: {}".format(model))

    # Data augmentation and normalization for training
    # Just normalization for validation
    logger.info("Initializing Datasets and Dataloaders...")
    logger.info("loading data {} from {}".format(cfgs.dataset, cfgs.data_path))
    dataloaders_dict = load_data(
        input_size=input_size,
        batch_size=cfgs.batch_size,
        data_path=cfgs.data_path,
        num_workers=cfgs.workers
    )
    # Detect if we have a GPU available
    device = torch.device(cfgs.device if torch.cuda.is_available() else "cpu")

    #  Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    param_log_info = ''
    if cfgs.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                param_log_info += "\t{}".format(name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_log_info += "\t{}".format(name)
    logger.info("Params to learn:\n" + param_log_info)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(
        params_to_update, lr=cfgs.lr, momentum=cfgs.momentum)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # NOTE: use neptune to track hyperparameters
    if cfgs.ml_track_framework == 'neptune':
        writer['config/model/model'] = cfgs.arch
        writer['config/model/feature_extract'] = cfgs.feature_extract
        writer['config/model/pretrained'] = cfgs.pretrained
        writer['config/dataset/num_classes'] = cfgs.num_classes
        writer['config/dataset/data_path'] = cfgs.data_path
        writer['config/dataset/dataset'] = cfgs.dataset
       
        writer['config/training/num_epochs'] = cfgs.epochs
        writer['config/training/batch_size'] = cfgs.batch_size
        writer['config/training/learning_rate'] = cfgs.lr
        writer['config/training/momentum'] = cfgs.momentum
        writer['config/training/weight_decay'] = cfgs.weight_decay
        writer['config/training/num_workers'] = cfgs.workers
        writer['config/training/weight_dir'] = cfgs.weight_dir

        writer['config/training/ml_track_framework'] = cfgs.ml_track_framework
        writer['config/training/device'] = cfgs.device

        writer['config/criterion'] = criterion.__class__.__name__
        writer['config/optimizer'] = optimizer_ft.__class__.__name__

    # Train and evaluate
    model_ft = train(model, dataloaders_dict, criterion, optimizer_ft,
                     logger, device, cfgs,
                     exp_tracker_framework='neptune', exp_tracker=writer
                     )
    mkdir(cfgs.weight_dir)
    torch.save(model_ft.state_dict(), os.path.join(
        cfgs.weight_dir, cfgs.arch) + '.ckpt')
    logger.info("model is saved at {}".format(os.path.abspath(
        os.path.join(cfgs.weight_dir, cfgs.arch) + '.ckpt')))


if __name__ == "__main__":
    main()

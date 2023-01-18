import copy
import time

import torch
from torchmetrics.functional import accuracy
from utils.meter import AverageMeter, ProgressMeter


def train(model, dataloaders, criterion, optimizer, logger, device, cfgs, exp_tracker_framework='tensorboard', exp_tracker=None):
    """a simple train and evaluate script modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html.

        Args:
            model (nn.Module): model to be trained.
            dataloaders (dict): should be a dict in the format of {'train': DataLoader, 'val': DataLoader}.
            device (Any): device.
            criterion (Any): loss function.
            optimizer (Any): optimizer.
            logger (Any): using logging.logger to print and log training information.
            print_freq (int): logging frequency.eg. 10 means logger will print information when 10 batches are trained or evaluated.
            num_epochs (int): training epochs
            tensorboard_plugin(Any): torch.utils.tensorboard.SummaryWriter
        Returns:
            model: trained model.
            
        """
    # Send the model to GPU
    model = model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(cfgs.epochs):
        # TODO (zhz): seperate train and val to two functions
        for phase in ['train', 'val']:
            # statistics
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.5f')
            top5 = AverageMeter('Acc@5', ':6.5f')
            progress = ProgressMeter(
                len(dataloaders[phase]),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            end = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                   
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # measure accuracy and record loss
                # TODO(zhz): revise the num classes
                acc1 = accuracy(outputs, labels, task="multiclass",
                                num_classes=cfgs.num_classes, top_k=1)
                acc5 = accuracy(outputs, labels, task="multiclass",
                                num_classes=cfgs.num_classes, top_k=5)
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1, inputs.size(0))
                top5.update(acc5, inputs.size(0))
                
                if phase == 'train':
                    if exp_tracker_framework == "neptune":
                        exp_tracker['train/loss'].append(loss)
                        exp_tracker['train/acc1'].append(acc1)
                        exp_tracker['train/acc5'].append(acc5)
                    elif exp_tracker_framework == "tensorboard":
                        exp_tracker.add_scalar('loss/train', loss, i)
                        exp_tracker.add_scalar('acc1/train', acc1, i)
                        exp_tracker.add_scalar('acc5/train', acc5, i)
                    elif exp_tracker_framework == "wandb":
                        exp_tracker.log({"train/loss": loss.item(), "train/acc1": acc1, "train/acc5": acc5})
                    else:
                        raise NotImplementedError
                
                else:
                    raise NotImplementedError

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % cfgs.print_freq == 0:
                    logger.info(progress.display(i))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

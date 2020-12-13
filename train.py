import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import Timer, store_labels
from vision.nets.ssd import MatchPrior
from vision.nets.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.utils.voc_dataset import VOCDataset
from vision.utils.multibox_loss import MultiboxLoss
from vision.utils.config import cfg
from vision.utils.data_preprocessing import TrainAugmentation, TestTransform

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

use_cuda = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

if use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def train(loader, net, criterion, optimizer, device, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(tqdm(loader)):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        # if i and i % debug_steps == 0:
      
    avg_loss = running_loss / len(loader)
    avg_reg_loss = running_regression_loss / len(loader)
    avg_clf_loss = running_classification_loss / len(loader)
    logging.info(
        f"Epoch: {epoch}, " +
        f"Average Loss: {avg_loss:.4f}, " +
        f"Average Regression Loss {avg_reg_loss:.4f}, " +
        f"Average Classification Loss: {avg_clf_loss:.4f}"
    )
    recode_loss_train.append(avg_loss)



def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)

    net = 'mb2-ssd-lite'

    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=cfg.mb2_width_mult)

    config = cfg

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []

    for dataset_path in args.datasets:
        dataset = VOCDataset(dataset_path, transform=train_transform,
                             target_transform=target_transform)
        label_file = os.path.join(cfg.checkpoint_dir, "voc-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)

        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, cfg.batch_size,
                              num_workers=1,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")

    val_dataset = VOCDataset(args.datasets, transform=test_transform, target_transform=target_transform, is_test=True)

    logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, cfg.batch_size,
                            num_workers=1,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = cfg.lr
    extra_layers_lr = cfg.lr

    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    timer.start("Load Model")

    logging.info(f"Init from base net {cfg.base_net}")
    net.init_from_base_net(cfg.base_net)

    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    logging.info(f"Learning rate: {cfg.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if cfg.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif cfg.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, cfg.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {cfg.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    fig = plt.figure()
    recode_loss_train = []
    recode_loss_test = []
    for epoch in range(last_epoch + 1, cfg.num_epochs):

        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, epoch=epoch)

        if epoch % cfg.validation_epochs == 0 or epoch == cfg.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            recode_loss_test.append(val_loss)
            if epoch % 10 == 0 or epoch == cfg.num_epochs - 1:
                model_path = os.path.join(cfg.checkpoint_dir, f"mb2-ssd-lite-{epoch}-Loss-{val_loss}.pth")
                net.save(model_path)
                logging.info(f"Saved model {model_path}")
            
    if not os.path.exists(cfg.save_loss_dir):
        os.mkdir(cfg.save_loss_dir)
    plt.plot(range(len(recode_loss_train)), recode_loss_train, label='Train')
    plt.plot(range(len(recode_loss_test)), recode_loss_test, label='Test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(cfg.save_loss_dir, 'loss.png'))
    # plt.plot(range(len(recode_loss_test)), recode_loss_test, label='Test')
    # plt.legend()
    # plt.savefig(os.path.join(cfg.save_loss_dir, 'loss_test.png'))
import os
import math
import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler

from model import *
from utils import train_one_epoch
import numpy as np
from tqdm import tqdm

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = os.path.join('../data/', "cancer_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = ShuffleNetV2(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            Eva_dict = {'TP': [0 for x in range(0, 5)],
                        'TN': [0 for x in range(0, 5)],
                        'FP': [0 for x in range(0, 5)],
                        'FN': [0 for x in range(0, 5)]}
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))

                predict_y = torch.max(outputs, dim=1)[1]

                for i in range(len(val_labels)):
                    predict = predict_y[i]
                    label = val_labels[i]

                    for j in range(5):
                        if predict == j:
                            if predict == label:
                                Eva_dict['TP'][j] = Eva_dict['TP'][j] + 1
                            else:
                                Eva_dict['FP'][j] = Eva_dict['FP'][j] + 1
                        else:
                            if predict == label:
                                Eva_dict['TN'][j] = Eva_dict['TN'][j] + 1
                            else:
                                Eva_dict['FN'][j] = Eva_dict['FN'][j] + 1

            print('TP:{}'.format(Eva_dict['TP']))
            print('TN:{}'.format(Eva_dict['TN']))
            print('FP:{}'.format(Eva_dict['FP']))
            print('FN:{}'.format(Eva_dict['FN']))
            TP = np.array(Eva_dict['TP'])
            TN = np.array(Eva_dict['TN'])
            FP = np.array(Eva_dict['FP'])
            FN = np.array(Eva_dict['FN'])
            mean_TP = np.mean(TP)
            mean_TN = np.mean(TN)
            mean_FP = np.mean(FP)
            mean_FN = np.mean(FN)
            accurate = (mean_TP + mean_TN) / (mean_TP + mean_TN + mean_FP + mean_FN)
            precision = mean_TP / (mean_TP + mean_FP)
            recall = mean_TP / (mean_TP + mean_FN)
            f1_score = 2 * recall * precision / (recall + precision)
            print('Mean Accurate: {}'.format(accurate))
            print('Mean Precision: {}'.format(precision))
            print('Mean Recall: {}'.format(recall))
            print('Mean F1_score: {}'.format(f1_score))

            tags = ["loss", "accuracy", "precision", "recall", "f1", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], accurate, epoch)
            tb_writer.add_scalar(tags[2], precision, epoch)
            tb_writer.add_scalar(tags[3], recall, epoch)
            tb_writer.add_scalar(tags[4], f1_score, epoch)
            tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--data-path', type=str,
                        default="/data/cancer_data")

    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x1_0
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import numpy as np
from tqdm import tqdm

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = os.path.join('../data/', "cancer_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 实例化训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 实例化验证数据集
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

    # 如果存在预训练权重则载入
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            Eva_dict = {'TP': [0 for x in range(0, 5)],
                        'TN': [0 for x in range(0, 5)],
                        'FP': [0 for x in range(0, 5)],
                        'FN': [0 for x in range(0, 5)]}
            val_bar = tqdm(val_loader)
            # val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))

                predict_y = torch.max(outputs, dim=1)[1]
                # print('predict_y {}'.format(predict_y))
                # print('val_labels {}'.format(val_labels))

                for i in range(len(val_labels)):
                    # print("prob: {:.3}".format(outputs[i][predict_y[i]]))
                    predict = predict_y[i]  # 预测值
                    label = val_labels[i]  # 真实值

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

                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
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



        # acc = evaluate(model=model,
        #                data_loader=val_loader,
        #                device=device)
        #
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # tags = ["loss", "accuracy", "learning_rate"]
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        #
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./checkpoints/shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

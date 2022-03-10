import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import MobileNetV2
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    batch_size = 16
    epochs = 101

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    
    image_path = os.path.join('../data/', "cancer_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = MobileNetV2(num_classes=5)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./checkpoints/mobilenet_v2.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './MobileNetV2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)


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
            tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
            tb_writer.add_scalar(tags[1], accurate, epoch)
            tb_writer.add_scalar(tags[2], precision, epoch)
            tb_writer.add_scalar(tags[3], recall, epoch)
            tb_writer.add_scalar(tags[4], f1_score, epoch)
            tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
            torch.save(net.state_dict(), "./weights/model-{}.pth".format(epoch))



    #             predict_y = torch.max(outputs, dim=1)[1]
    #             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    #             val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
    #                                                        epochs)
    #     val_accurate = acc / val_num
    #     print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #           (epoch + 1, running_loss / train_steps, val_accurate))

    #     if val_accurate > best_acc:
    #         best_acc = val_accurate
    #         torch.save(net.state_dict(), save_path)

    # print('Finished Training')


if __name__ == '__main__':
    main()

import os
import json

import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model import resnet34
import numpy as np

import pandas as pd

from utils import analysis

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load dataset
    image_path = os.path.join('../data/', "cancer_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images fot validation.".format(val_num))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./weights/model-58.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))


    label_list = []
    likelihood_list = []
    pred_list = []
    val_label_list = []
    # prediction
    model.eval()
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device)) #  输出16*5的矩阵，用来
            for i in range(len(val_labels)):
                val_label = val_labels[i]
                output = outputs[i].unsqueeze(0)
                output_max = output.max(dim=1)[0].repeat(5, 1).t()
                output_min = output.min(dim=1)[0].repeat(5, 1).t()
                output_normal = (output - output_min) / (output_max - output_min)
                pred_info = output_normal / output_normal.sum(axis=1).repeat(5, 1).t()
                # pred_info = torch.softmax(output, 1).cpu().numpy()
                # 获取可能性最大的标签
                # label = torch.softmax(output,1).cpu().numpy().argmax()
                label = output.cpu().numpy().argmax()
                label_list.append(label)                     #
                # 获取可能性最大的值（即概率）
                # likelihood = torch.softmax(output,1).cpu().numpy().max()
                likelihood = output.cpu().numpy().max()
                likelihood_list.append(likelihood)           #
                pred_list.append(pred_info.tolist()[0])   #
                val_label_list.append(val_label.numpy().tolist())

        predict_data = np.array(pred_list).reshape((-1, 5))

        analysis.compare(val_label_list, predict_data)



        # label_names = ["CA", "CE", "HSIL", "LSIL", "Normal"]
        #
        # df_pred = pd.DataFrame(data=pred_list, columns=label_names)
        #
        # df_pred.to_csv('pred_result_3.csv', encoding='gbk', index=False)
        # print("Done!")

if __name__ == '__main__':
    main()

import os
import json
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pyheatmap.heatmap import HeatMap
from model import shufflenet_v2_x1_0
import numpy as np


def apply_heatmap(image,data):
    '''image是原图，data是坐标'''
    '''创建一个新的与原图大小一致的图像，color为0背景为黑色。这里这样做是因为在绘制热力图的时候如果不选择背景图，画出来的图与原图大小不一致（根据点的坐标来的），导致无法对热力图和原图进行加权叠加，因此，这里我新建了一张背景图。'''
    background = Image.new("RGB", (image.shape[1], image.shape[0]), color=0)
    # 开始绘制热度图
    hm = HeatMap(data)
    hit_img = hm.heatmap(base=background, r = 100) # background为背景图片，r是半径，默认为10
    # ~ plt.figure()
    # ~ plt.imshow(hit_img)
    # ~ plt.show()
    #hit_img.save('out_' + image_name + '.jpeg')
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)#Image格式转换成cv2格式
    overlay = image.copy()
    alpha = 0.5 # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1) # 设置蓝色为热度图基本色蓝色
    image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0) # 将背景热度图覆盖到原图
    image = cv2.addWeighted(hit_img, alpha, image, 1-alpha, 0) # 将热度图覆盖到原图


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_GaussianBlur = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         # transforms.RandomAffine(60),
         transforms.GaussianBlur(3),
         # transforms.Grayscale(num_output_channels=1),
         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         transforms.ToPILImage()
         ])

    transform_Grayscale = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         # transforms.RandomAffine(60),
         transforms.GaussianBlur(3),
         transforms.Grayscale(num_output_channels=1),
         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         transforms.ToPILImage()
         ])

    transform_Normalize = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         # transforms.RandomAffine(60),
         transforms.GaussianBlur(3),
         transforms.Grayscale(num_output_channels=3),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         transforms.ToPILImage()
         ])



    # load image
    img_path = "img.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    img.save("org.jpg")
    # [N, C, H, W]
    GaussianBlur = transform_GaussianBlur(img)
    GaussianBlur.save("GaussianBlur.jpg")

    Grayscale = transform_Grayscale(img)
    Grayscale.save("Grayscale.jpg")

    Normalize = transform_Normalize(img)
    Normalize.save("Normalize.jpg")

    # plt.imshow(img)
    # plt.show()


"""
SE_before = model.stage4._modules["3"].branch2._modules["3"].weight       232.1.3.3
SE_after = model.stage4._modules["3"].branch2._modules["6"].weight   232.232.1.1
SE_after = SE_after.squeeze(2)    232.232.1
SE_after = SE_after.squeeze(2)    232.232
SE_after = SE_after.unsqueeze(0)  1.232.232
ToPILImage = transforms.ToPILImage()
SE_after_PIL = ToPILImage(SE_after)
"""

if __name__ == '__main__':
    main()
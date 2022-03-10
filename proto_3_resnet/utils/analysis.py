from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd # pip install pandas

# true_label  # list 698
# predict_data   #  dataframe 698,5   numpy 数组

def compare(true_label, predict_data):
    print(true_label)
    print(predict_data)

    predict_label = predict_data.argmax(axis=1)
    print("predict_label = ",predict_label )
    predict_score = predict_data.max(axis=1)

    print("predict_score = ",predict_score )

    # 精度，准确率， 预测正确的占所有样本种的比例
    accuracy = accuracy_score(true_label, predict_label)
    print("精度: ",accuracy)

    # 查准率P（准确率），precision(查准率)=TP/(TP+FP)
    precision = precision_score(true_label, predict_label, labels=None, pos_label=1, average='macro') # 'micro', 'macro', 'weighted'
    print("查准率P: ",precision)

    # 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
    recall = recall_score(true_label, predict_label, average='macro') # 'micro', 'macro', 'weighted'
    print("召回率: ",recall)

    # F1-Score
    f1 = f1_score(true_label, predict_label, average='macro')     # 'micro', 'macro', 'weighted'
    print("F1 Score: ",f1)


    '''
    混淆矩阵
    '''
    label_names = ["CA", "CE", "HSIL", "LSIL", "Normal"]
    label_names_re = ["Normal", "LSIL", "HSIL", "CE", "CA"]
    confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])
    sum_info = np.sum(confusion,axis=1)
    sum_infos = np.concatenate((sum_info,sum_info,sum_info,sum_info,sum_info))
    sum_infos = sum_infos.reshape(5,5)
    confusion_out  = confusion / sum_infos
    confusion_out = np.flip(confusion_out,axis=0)
    pd.DataFrame(data=confusion_out).to_csv('./save_confusion.csv')
    # print("混淆矩阵: \n",confusion)
    # plt.figure()
    plt.matshow(confusion_out, cmap=plt.cm.Blues)   # Greens, Blues, Oranges, Reds
    plt.colorbar()
    for i in range(len(confusion_out)):
        for j in range(len(confusion_out)):
            infos = ('%.2f' % confusion_out[i, j])
            plt.annotate(infos, xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(range(len(label_names)), label_names)
    plt.yticks(range(len(label_names)), label_names_re)
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.title("Confusion Matrix")
    # plt.subplots_adjust(left=0.15,right=0.95,wspace=0.25,hspace=0.25,bottom=0.15,top=0.95)
    # plt.savefig("Confusion_Matrix.png")
    plt.show()


    '''
    ROC曲线（多分类）
    '''
    n_classes = len(label_names)
    binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])

    # 读取预测结果

    predict_score = predict_data

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarize_predict[:,i], [socre_i[i] for socre_i in predict_score])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)


    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.1,right=0.95,wspace=0.25,hspace=0.25,bottom=0.1,top=0.95)
    plt.grid()
    plt.savefig('ROC.jpg')
    plt.show()
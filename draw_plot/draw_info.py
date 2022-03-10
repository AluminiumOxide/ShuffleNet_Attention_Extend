import pandas as pd
import matplotlib.pyplot as plt


def draw_split_plot(network_list):
    for network in network_list:
        info_all = pd.read_excel('info_all_backup.xlsx',sheet_name=network)
        print(info_all)
        epoch=info_all['Step']
        accurate = info_all['accurate']
        loss = info_all['loss']

        plt.figure()
        plt.title('{} Accurate'.format(network))
        plt.xlabel('Epoch')
        plt.ylabel('Accurate')
        plt.xlim([0, 100])  # x轴边界
        # plt.ylim([0,1])  # y轴边界
        plt.plot(accurate,label=network)
        plt.legend(loc = 'upper left')
        plt.grid()
        plt.savefig('{} Accurate'.format(network), bbox_inches='tight')
        # plt.show()

        plt.figure()
        plt.title('{} Loss'.format(network))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim([0, 100])  # x轴边界
        # plt.ylim([0,1])  # y轴边界
        plt.plot(loss,label=network)
        plt.legend(loc = 'upper right')
        plt.grid()
        plt.savefig('{} Loss'.format(network), bbox_inches='tight')
        # plt.show()


def draw_mix_plot(network_list):
    for network in network_list:
        info_all = pd.read_excel('info_all_backup.xlsx',sheet_name=network)
        print(info_all)
        epoch=info_all['Step']
        accurate = info_all['accurate']
        loss = info_all['loss']



        fig, ax1 = plt.subplots()

        # 产生一个ax1的镜面坐标
        ax2 = ax1.twinx()

        ax1.plot(accurate,'-',label='Accurate')
        plt.legend(loc='upper left')
        ax2.plot(loss,label='Loss')
        plt.legend(loc='upper left')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accurate')
        ax2.set_ylabel('Loss')

        plt.title('{}'.format(network))
        plt.xlim([0, 100])  # x轴边界
        plt.legend(loc = 'upper left')

        plt.grid()
        plt.tight_layout()
        # plt.savefig('{}_mix'.format(network), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    network_list = ['VGG-16', 'GoogleNet', 'ResNet', 'MobileNet', 'DenseNet']
    # draw_split_plot(network_list)
    draw_mix_plot(network_list)

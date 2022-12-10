import math
import torch
from torch import nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from torchvision import datasets,transforms 


# Modified_SPPLayer是解决到最后的时候，
# 由于pad的size会大于kernel_size/2，因此pooling会报错
# pad should be smaller than half of kernel size,
# but got padW = 1, padH = 3, kW = 1, kH = 2 at /pytorch/aten/src/THNN/generic/SpatialDilatedMaxPooling.c:35
class SPPLayer(torch.nn.Module):

    # 定义Layer需要的额外参数（除Tensor以外的）
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    # forward()的参数只能是Tensor(>=0.4.0) Variable(< 0.4.0)
    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        level = 1
        #         print(x.size())
        for i in range(self.num_levels):
            level <<= 1

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))  # kernel_size = (h, w)
            padding = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data with padding
            #  class torch.nn.ZeroPad2d(padding)[source]
            #
            #     Pads the input tensor boundaries with zero.
            #
            #     For N`d-padding, use :func:`torch.nn.functional.pad().
            #     Parameters:   padding (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries.
            # If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)
            zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new, w_new = x_new.size()[2:]

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten

# 为了节约时间, 我们测试时只测试前2000个
#test_x = torch.unsqueeze(train_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#test_y = test_data.test_labels[:2000]
def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
        )
        self.spp_layer = SPPLayer(3)
        self.out = nn.Linear(2688, 10)   # calc_auto(3,10)210

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.spp_layer(x)
        output = self.out(x)
        return output

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    return rights, len(labels) 

loss_data = []
def spp_train():
    torch.manual_seed(1)    # reproducible
    
    # Hyper Parameters
    EPOCH = 5           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 50
    LR = 0.001          # 学习率
    DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False
    
    #1.加载数据集
    train_dataset = datasets.MNIST(root='./data',  
                                train=True,   
                                transform=transforms.ToTensor(),  
                                download=True) 
    
    # 测试集
    test_dataset = datasets.MNIST(root='./data', 
                               train=False, 
                               transform=transforms.ToTensor())
    
    # 构建batch数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)    
    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted    
    # training and testing
    for epoch in range(EPOCH):
        train_rights = [] 
        for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
            net.train()                   
            output = net(data) 
            loss = loss_func(output, target) 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            right = accuracy(output, target) 
            train_rights.append(right) 

            if batch_idx % 50 == 0:
                net.eval()
                val_rights = [] 
                
                for (data, target) in test_loader:
                    output = net(data) 
                    right = accuracy(output, target) 
                    val_rights.append(right)    
                loss_data.append(loss.data)
                #准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                
                print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                    epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.data, 
                    100. * train_r[0].numpy() / train_r[1], 
                    100. * val_r[0].numpy() / val_r[1]))
    torch.save(net,'data\\CNN.pt')    

if __name__ == '__main__': 
    spp_train()       
    data_path = "data\\Test_Numbers"
    net = torch.load("data/CNN.pt")
    net.eval()
    img_files = [os.path.join(data_path, path) \
                 for path in os.listdir(data_path) \
                 if '.jpg' in path]
    correct = 0
    total = len(img_files)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    X=np.arange(len(loss_data))
    plt.plot(X,loss_data)
    plt.ylabel("损失")
    plt.xlabel("迭代次数")
    plt.xticks(X[:-1:10],X[:-1:10]*555,rotation = 60)
    
    for img_file in img_files:
        test_img = cv.imread(img_file)
        gray = cv.cvtColor(test_img,cv.COLOR_BGR2GRAY)
        gray = np.ones(gray.shape)*255 - gray
        test_tensor = transforms.ToTensor()(gray)
        test_tensor = test_tensor.float()     
        test_output = net(test_tensor.unsqueeze(0))
        pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print('prediction number:',pred)
        print('real number:', img_file[-5])
        if pred == eval(img_file[-5]):
            correct  = correct + 1
            
    print('-'*40)
    print("accuracy:{:.4f}%".format(100* correct / total))
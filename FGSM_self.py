import torch
import torch.nn as nn
from torchvision import datasets,transforms 
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

device=torch.device('cuda'if torch.cuda.is_available() else"cpu")
batch_size=10
#定义预处理
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
transform_origin=transforms.Compose([
    transforms.ToTensor()
])

os.makedirs('origin input png', exist_ok=True)
os.makedirs('adversarial perturbation png', exist_ok=True)
os.makedirs('adversarial sample png', exist_ok=True)
#定义测试集
testset=datasets.CIFAR10(root='./Datasets',train=False,download=True,transform=transform)
originset=datasets.CIFAR10(root='./Datasets',train=False,download=True,transform=transform_origin)

#定义loader
num_origin=1
testloader=DataLoader(testset,batch_size=batch_size,shuffle=False)
originloader=DataLoader(originset,batch_size=batch_size,shuffle=False)

#保存原始图像
for data in originloader:
    inputs,label=data[0].to(device),data[1].to(device)
    orig_img = Image.fromarray((inputs.cpu().detach().numpy()[0].transpose(1, 2, 0) * 255).astype('uint8'))
    orig_img.save(f'origin input png/original_input{num_origin}.png')
    num_origin+=1

#模型定义
model=resnet152(pretrained=False)
num_str=model.fc.in_features
model.fc=nn.Sequential(
    nn.Linear(num_str,10),
    nn.Sigmoid()
)

#加载模型
model.load_state_dict(torch.load('resnet152_cifar10_best_2.pth'))
model=model.to(device)

judge=model.eval()
loss=0
correct=0
total=0
epoch=1
acc_list=[]

#开始攻击
for epsilon in np.linspace(0, 1, 100):
    num = 1
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs.requires_grad_(True)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        #计算datagrad
        model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data


        #保存图像
        if epoch>=2:
            # 保存对抗性扰动图像
            pert_img = Image.fromarray(
            ((epsilon * torch.sign(data_grad)).cpu().detach().numpy()[0].transpose(1, 2, 0) * 255).astype('uint8'))
            pert_img.save(f'adversarial perturbation png/adversarial_perturbation{epoch}-{num}.png')

            # 保存对抗性样本图像
            adv_img = Image.fromarray(
                ((inputs + epsilon * torch.sign(data_grad)).cpu().detach().numpy()[0].transpose(1, 2, 0) * 255).astype(
                    'uint8'))
            adv_img.save(f'adversarial sample png/adversarial_sample{epoch}-{num}.png')
            num += 1
        #加噪重新预测
        inputs = inputs + epsilon * torch.sign(data_grad)
        out = judge(inputs)
        _, predicted = torch.max(out.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = correct / total
    acc_list.append({epoch:test_acc})
    print(f'epoch{epoch}: acc:{test_acc*100}%')
    epoch+=1

# 分辨率参数-dpi，画布大小参数-figsize
# plt.figure(dpi=300,figsize=(24,8))
plt.title("准确率")  # ("LassoRegression")#("LinearRegression预测结果")
plt.rc('font', family='SimHei');
plt.rc('font', size=15)
# 提取字典的键和值
x = list(acc_list.keys())/100
y = list(acc_list.values())
# 绘制折线图
plt.plot(x, y)
# 添加标题和标签plt.title('折线图')
plt.xlabel('步长/ε')
plt.ylabel('准确率/acc')
# plt.legend(loc=1)
#    plt.savefig(path + "/" + name + r"每日预测准确率.png")

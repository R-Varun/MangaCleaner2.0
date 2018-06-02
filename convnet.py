import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import shutil

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def resize(image, width = 30, height = 30):
	r = image.resize((width,height))
	r = np.asarray(r)

	# r = r.flatten()
	return r


def grabData():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    neg = os.path.join(dir_path, "neg_characters", "img")
    pos = os.path.join(dir_path, "pos_characters", "img")

    width = 20
    height = 20

    neg_imgs = list(map(lambda x: resize(Image.open(os.path.join(neg, x)).convert('RGB')), os.listdir(neg)))
    pos_imgs = list(map(lambda x: resize(Image.open(os.path.join(pos, x)).convert('RGB')), os.listdir(pos)))
    np.random.shuffle(neg_imgs)
    np.random.shuffle(pos_imgs)

    hn = int(len(neg_imgs) // 2)
    hp = int(len(pos_imgs) // 2)

    trainSet = neg_imgs[:hn] + pos_imgs[:hp]
    train_data = np.array(trainSet)

    print(len(train_data[0][0][0]))

    target_train = [0 for x in range(hn)] + [1 for x in range(hp)]

    target_train = np.array(target_train)

    testSet = neg_imgs[hn:] + pos_imgs[hp:]

    test_data = np.array(testSet)
    target_train2 = [0 for x in range(len(neg_imgs) - hn)] + [1 for x in range(len(pos_imgs) - hp)]

    target_train2 = np.array(target_train2)

    t = transforms.ToTensor()

    data = list(map(lambda x : t(x), train_data))

    data2 = list(map(lambda x : t(x), test_data))


    return (data, target_train, data2, target_train2)



import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


dir_path = os.path.dirname(os.path.realpath(__file__))
neg = os.path.join(dir_path, "neg_characters", "img")
pos = os.path.join(dir_path, "pos_characters", "img")

# class Inception(nn.Module):
#     def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
#         super(Inception, self).__init__()
#         # 1x1 conv branch
#         self.b1 = nn.Sequential(
#             nn.Conv2d(in_planes, n1x1, kernel_size=1),
#             nn.BatchNorm2d(n1x1),
#             nn.ReLU(True),
#         )
#
#         # 1x1 conv -> 3x3 conv branch
#         self.b2 = nn.Sequential(
#             nn.Conv2d(in_planes, n3x3red, kernel_size=1),
#             nn.BatchNorm2d(n3x3red),
#             nn.ReLU(True),
#             nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n3x3),
#             nn.ReLU(True),
#         )
#
#         # 1x1 conv -> 5x5 conv branch
#         self.b3 = nn.Sequential(
#             nn.Conv2d(in_planes, n5x5red, kernel_size=1),
#             nn.BatchNorm2d(n5x5red),
#             nn.ReLU(True),
#             nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n5x5),
#             nn.ReLU(True),
#             nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n5x5),
#             nn.ReLU(True),
#         )
#
#         # 3x3 pool -> 1x1 conv branch
#         self.b4 = nn.Sequential(
#             nn.MaxPool2d(3, stride=1, padding=1),
#             nn.Conv2d(in_planes, pool_planes, kernel_size=1),
#             nn.BatchNorm2d(pool_planes),
#             nn.ReLU(True),
#         )
#
#     def forward(self, x):
#         y1 = self.b1(x)
#         y2 = self.b2(x)
#         y3 = self.b3(x)
#         y4 = self.b4(x)
#         return torch.cat([y1,y2,y3,y4], 1)
#
#
# class GoogLeNet(nn.Module):
#     def __init__(self):
#         super(GoogLeNet, self).__init__()
#         self.pre_layers = nn.Sequential(
#             nn.Conv2d(3, 192, kernel_size=3, padding=1),
#             nn.BatchNorm2d(192),
#             nn.ReLU(True),
#         )
#
#         self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
#         self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
#
#         self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
#
#         self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
#         self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
#         self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
#         self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
#         self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
#
#         self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
#
#         self.avgpool = nn.AvgPool2d(8, stride=1)
#         self.linear = nn.Linear(1024, 10)
#
#     def forward(self, x):
#         out = self.pre_layers(x)
#         out = self.a3(out)
#         out = self.b3(out)
#         out = self.maxpool(out)
#         out = self.a4(out)
#         out = self.b4(out)
#         out = self.c4(out)
#         out = self.d4(out)
#         out = self.e4(out)
#         out = self.maxpool(out)
#         out = self.a5(out)
#         out = self.b5(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = Net()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), .0001)

transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.5, hue=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root="./train", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root="./test", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=True, num_workers=4)


criterion = nn.CrossEntropyLoss()

cuda = torch.cuda.is_available()

# cuda = False
if cuda:
    model = model.cuda()

def train(epoch, save=False):
    model.train()
    total_correct = 0
    train_loader = trainloader
    for batch_idx, (data, target) in enumerate(train_loader):
        print(target)
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        total_correct += correct
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc: {:.2f}%/{:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0],
                       100. * correct / train_loader.batch_size,
                       100. * total_correct / ((batch_idx + 1) * train_loader.batch_size)))

    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'arch': "ye",
    #     'state_dict': model.state_dict(),
    #     'best_prec1': 0,
    #     'optimizer': optimizer.state_dict(),
    # }, True)
    # print("SAVED")


global global_model
global_model = None
def loadModel():
    global global_model
    if global_model == None:
        if os.path.isfile('models/model_best.pth.tar'):
            checkpoint = None
            cuda = torch.cuda.is_available()
            # cuda = False
            if cuda:
                checkpoint = torch.load('models/model_best.pth.tar')
            else:
                torch.load('my_file.pt', map_location=lambda storage, loc: storage)

            nm = Net()
            nm.load_state_dict(checkpoint['state_dict'])
            op = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
            op.load_state_dict(checkpoint['optimizer'])
            if cuda:
                nm = nm.cuda()
            global_model = nm, op

            return nm, op
    else:
        return global_model


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = testloader

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, test_acc


imsize = 256
loader = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def image_loader(image):
    """load image, returns cuda tensor"""
    image = image.convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU


def predict(image):

    n, o = loadModel()
    test = image_loader(image)

    outputs = n(test)
    _, predicted = torch.max(outputs.data, 1)
    return int(predicted[0])

def main():
    for i in range(200):
        train(i)
    # print("testing")
    # n, o = loadModel()
    # n= n.cuda()
    # print(test(n))





if __name__ == "__main__":
    main()
import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.optim as optim
import shutil

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torchvision.models as models

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

dir_path = os.path.dirname(os.path.realpath(__file__))
neg = os.path.join(dir_path, "neg_characters", "img")
pos = os.path.join(dir_path, "pos_characters", "img")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), .0001)

transform = transforms.Compose([transforms.Resize((34, 34)),
                                transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.5, hue=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder(root="./train", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root="./test", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=2)


criterion = nn.CrossEntropyLoss()


def train(epoch, save=False):
    model.train()
    total_correct = 0

    for epoch in range(epoch):  # loop over the dataset multiple times
        print("CURRENT EPOCH " + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 3000 == 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            if save:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': "ye",
                    'state_dict': model.state_dict(),
                    'best_prec1': 0,
                    'optimizer': optimizer.state_dict(),
                }, True)
        test()

    # print('Finished Training')
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': "ye",
        'state_dict': model.state_dict(),
        'best_prec1': 0,
        'optimizer': optimizer.state_dict(),
    }, True)



def loadModel():
    if os.path.isfile('model_best.pth.tar'):
        checkpoint = torch.load('model_best.pth.tar')
        nm = Net()
        nm.load_state_dict(checkpoint['state_dict'])
        op = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        op.load_state_dict(checkpoint['optimizer'])
        return nm, op
    else:
        return None


def test():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


imsize = 256
loader = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def image_loader(image):
    """load image, returns cuda tensor"""
    image = image.convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU


def predict(image):
    n, o = loadModel()
    test = image_loader(image)

    outputs = n(test)
    _, predicted = torch.max(outputs.data, 1)
    return int(predicted[0])

def main():
    train(200, save=True)

    # global model
    # global optimizer
    #
    # model, optimizer = loadModel()
    #
    test()

    pass



if __name__ == "__main__":
    main()
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

IMAGE_DIR = "./Region/complete/raw"
BBOX_DIR = "./Region/complete/bbox"

SAVE_POS_DIR = "./temp/pos"

SAVE_NEG_DIR = "./temp/neg"


def save_checkpoint(state, is_best, filename='checkpoint_RPN.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_RPN_best.pth.tar')


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import utils


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(7744, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 3)

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

image_list = os.listdir(IMAGE_DIR)
annotations = os.listdir(BBOX_DIR)


def RPN_trainloader():
	# Skip ever 100 no intersection
	pos_count = 0
	for i, image_loc in enumerate(image_list):
		print("NEW IMAGE")
		name = image_loc[:image_loc.index(".")]

		cur_image = np.asarray(Image.open(os.path.join(IMAGE_DIR, image_loc)))
		bboxes = json.loads(open(os.path.join(BBOX_DIR, name + ".json"), "r").read())

		rectArr = list(map(lambda x: utils.Rectangle(int(min(x[0], x[1])),
													 int(min(x[2], x[3])),
													 int(max(x[0], x[1])),
													 int(max(x[2], x[3]))), bboxes))

		image = cur_image
		sizes = [(image.shape[1] // 15, image.shape[0] // 20), (image.shape[1] // 30, image.shape[0] // 30),
				 (image.shape[1] // 20, image.shape[0] // 15), (image.shape[1] // 30, image.shape[0] // 40)]


		for width, height in sizes:
			for window in utils.sliding_window(image, (10, 10), windowSize=(width, height)):
				curRect = utils.Rectangle(window[0], window[1], window[0] + width, window[1] + height)
				area = calculateIntersectArea(curRect, rectArr)
				ratio = area / curRect.area
				label = 0
				if ratio > .3:
					label = 1
				if ratio > .7:
					label = 2

				if label == 0:
					pos_count += 1
					if pos_count % 10 != 0:
						continue
				# print(area, curRect.area)
				# if area > 500:
				#     plt.imshow(window[2])
				#     plt.show()
				img = image_loader(Image.fromarray(window[2]))



				# pilTrans = transforms.ToPILImage()
				# pilImg = pilTrans(img.cpu())
				# plt.imshow(np.asarray(pilImg))
				# plt.show()
				yield (img, label)


def calculateIntersectArea(rectangle, rectArr):
	area = 0
	for i in rectArr:
		if rectangle.intersects(i):
			area += rectangle.intersection(i).area
	return area


def make_one_hot(labels, C=2):
	'''
	Converts an integer label torch.autograd.Variable to a one-hot Variable.

	Parameters
	----------
	labels : torch.autograd.Variable of torch.cuda.LongTensor
		N x 1 x H x W, where N is batch size.
		Each value is an integer representing correct classification.
	C : integer.
		number of classes in labels.

	Returns
	-------
	target : torch.autograd.Variable of torch.cuda.FloatTensor
		N x C x H x W, where C is class number. One-hot encoded.
	'''
	one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
	target = one_hot.scatter_(1, labels.data, 1)

	target = Variable(target)

	return target


def train(epoch, save=False):
	model.train()
	total_correct = 0
	# train_loader = trainloader
	counter = 0
	total_pos = 0
	count_pos = 0
	for data, target in RPN_trainloader():
		ut = target

		# nt = torch.LongTensor(1, 3)
		# nt.zero_()
		# nt[0][target] = 1
		# target = nt
		# print(target)
		target = torch.LongTensor([target])
		# nt[0] = target
		# target = nt
		# if ut != 0:
		#     print(target)


		data, target = data, Variable(target)
		if cuda:
			data, target = data.cuda(), target.cuda()

		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		pred = output.data.max(1)[1]  # get the index of the max log-probability

		correct = pred.eq(target.data.view_as(pred)).cpu().sum()
		# correct = pred.eq(target.data.view_as(pred)).cpu().sum()
		if (ut != 0):
			count_pos += correct
			total_pos += 1
			# print(pred)

		total_correct += correct
		if counter % 100 == 0:
			print('Train Epoch: {:.2f}% CORRECT POS {:.2f}% NUM POS {}/{} LOSS {}'.format(
				100. * total_correct / ((counter + 1)), 100 * (count_pos / (total_pos + 1)), count_pos, total_pos,
				loss.data[0]))

		if counter % 1000 == 0:
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': "ye",
				'state_dict': model.state_dict(),
				'best_prec1': 0,
				'optimizer': optimizer.state_dict(),
			}, True)
			print("SAVED")

		counter += 1

	save_checkpoint({
		'epoch': epoch + 1,
		'arch': "ye",
		'state_dict': model.state_dict(),
		'best_prec1': 0,
		'optimizer': optimizer.state_dict(),
	}, True)
	print("SAVED")


global global_model
global_model = None


def loadModel():
	global global_model
	if global_model == None:
		if os.path.isfile('models/model_RPN_best.pth.tar'):
			checkpoint = None
			cuda = torch.cuda.is_available()
			# cuda = False
			if cuda:
				checkpoint = torch.load('models/model_RPN_best.pth.tar')
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
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		if int(pred[0]) == 2:
			pred[0] = 1
		if int(pred[0]) == 1:
			pred[0] = 0
		correct += pred.le(target.data.view_as(pred)).cpu().sum()

	test_acc = 100. * correct / len(test_loader.dataset)
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		test_acc))

	return test_loss, test_acc


imsize = 256
loader = transforms.Compose([transforms.Resize((100, 100)),
							 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

def image_loader(image):
	"""load image, returns cuda tensor"""
	image = image.convert("RGB")
	image = loader(image).float()
	image = Variable(image, requires_grad=True)
	image = image.unsqueeze(0)
	image = image.cuda()
	# image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
	return image  # assumes that you're using GPU


def predict(image):
	n, o = loadModel()
	test = image_loader(image)

	outputs = n(test)
	_, predicted = torch.max(outputs.data, 1)
	# print(predicted)
	return int(predicted[0])


def main():
	# for i in range(10):
	# 	train(i)
	# print("testing")
	n, o = loadModel()
	n = n.cuda()
	print(test(n))


if __name__ == "__main__":
	main()

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

from sklearn.neural_network import MLPClassifier


from sklearn import datasets, svm, metrics

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def resize(image, width = 30, height = 30):
	r = image.resize((width,height))
	r = np.asarray(r)

	r = r.flatten()
	return r

def create_MLP_classifier():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	neg = os.path.join(dir_path, "neg_characters", "img")
	pos = os.path.join(dir_path, "pos_characters", "img")

	width = 20
	height = 20

	neg_imgs = list(map(lambda x: resize(Image.open(os.path.join(neg, x)).convert('L')), os.listdir(neg)))
	pos_imgs = list(map(lambda x: resize(Image.open(os.path.join(pos, x)).convert('L')), os.listdir(pos)))

	hn = int(len(neg_imgs) // 2)
	hp = int(len(pos_imgs) // 2)

	trainSet = neg_imgs[:hn] + pos_imgs[:hp]
	train_data = np.array(trainSet)
	target_train = [0 for x in range(hn)] + [1 for x in range(hp)]
	print(target_train)
	target_train = np.array(target_train)

	testSet = neg_imgs[hn:] + pos_imgs[hp:]

	test_data = np.array(testSet)
	target_train2 = [0 for x in range(len(neg_imgs) - hn)] + [1 for x in range(len(pos_imgs) - hp)]

	target_train2 = np.array(target_train2)

	data = train_data
	data2 = test_data

	clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
						beta_1=0.9, beta_2=0.999, early_stopping=False,
						epsilon=1e-08, hidden_layer_sizes=(100, ), learning_rate='constant',
						learning_rate_init=0.001, max_iter=2000, momentum=0.9,
						nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
						solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
						warm_start=False)


	clf.fit(data, target_train)

	# Now predict the value of the digit on the second half:
	expected = target_train2
	predicted = clf.predict(data2)

	print("Classification report for classifier %s:\n%s\n"
		  % (clf, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	while True:
		lol = input("input image path")
		i = Image.open(os.path.join(dir_path, lol)).convert('L')
		i = np.asarray(i)
		plt.imshow(i)
		plt.show()
		trial = resize(Image.open(os.path.join(dir_path, lol)).convert('L')).reshape(1, -1)

		print(clf.predict(trial))



def create_classifier():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	neg = os.path.join(dir_path, "neg_characters", "img")
	pos = os.path.join(dir_path, "pos_characters", "img")

	width = 20
	height = 20

	neg_imgs = list(map( lambda x : resize(Image.open(os.path.join(neg, x)).convert('L')), os.listdir(neg)))
	pos_imgs = list(map( lambda x : resize(Image.open(os.path.join(pos, x)).convert('L')), os.listdir(pos)))

	hn = int(len(neg_imgs) // 2)
	hp = int(len(pos_imgs) // 2)

	trainSet = neg_imgs[:hn] + pos_imgs[:hp]
	train_data = np.array(trainSet)
	target_train = [0 for x in range(hn) ] + [1 for x in range(hp)]
	print(target_train)
	target_train = np.array(target_train)


	testSet = neg_imgs[hn:] + pos_imgs[hp:]

	test_data = np.array(testSet)
	target_train2 = [0 for x in range(len(neg_imgs) - hn)] + [1 for x in range(len(pos_imgs) - hp)]


	target_train2 = np.array(target_train2)

	data = train_data
	data2 = test_data


	print(data)
	# print(len(data[0]))
	# print(len(data))
	# Create a classifier: a support vector classifier
	classifier = svm.SVC(gamma=0.001, kernel='poly',probability=True, verbose= True)

	# We learn the digits on the first half of the digits
	classifier.fit(data, target_train)

	# Now predict the value of the digit on the second half:
	expected = target_train2
	predicted = classifier.predict(data2)

	print("Classification report for classifier %s:\n%s\n"
		  % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	while True:
		lol = input("input image path")
		i = Image.open(os.path.join(dir_path, lol)).convert('L')
		i = np.asarray(i)
		plt.imshow(i)
		plt.show()
		trial = resize(Image.open(os.path.join(dir_path, lol)).convert('L')).reshape(1, -1)

		print(classifier.predict(trial))

create_MLP_classifier()
# create_classifier()


def setupHAAR():
	dir_path = os.path.dirname(os.path.realpath(__file__))

	neg = os.path.join(dir_path, "neg_characters")
	pos = os.path.join(dir_path, "pos_characters")


	nb = os.path.join(neg, "bg.txt")
	pb = os.path.join(pos, "bg.txt")
	print(neg)
	print(os.listdir(neg))

	with open(nb, 'w') as f:
		for i in os.listdir(os.path.join(neg, "img")):
			f.write("img/"+ i + "\n")

	with open(pb, 'w') as f:
		for i in os.listdir(os.path.join(pos, "img")):
			f.write("img/" + i + "\n")

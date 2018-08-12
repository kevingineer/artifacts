import cv2
# cv2 is OpenCV module that allows to read, display, and save an image
# read an image via function cv2.imread:
#   cv2.IMREAD_COLOR('image.jpg') - Loads a color image.Any transparency of image will be neglected. It is the default flag.
#   cv2.IMREAD_GRAYSCALE - Loads image in grayscale mode
#   cv2.IMREAD_UNCHANGED - Loads image as such including alpha channel
#   cv2.imshow('image', img) - image is the window name, second is the image
#   cv2.waitKey() - argument is time in milliseconds. 
#   cv2.destroyAllWindows() / cv2.destroyWindows()
#   cv2.imwrite('messigray.png',img) to save an image.
# Note Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.
#   example: img = cv2.imread('messi5.jpg',0)
# Warning Even if the image path is wrong, it won’t throw any error, but print img will give you None
# Note on cv2.waitkey():  .."esc" 27. "control" 19. "Shift" 17, "CapsLock" 18, "Alt" 20. "NumLock" 21.
# if k == 27 or k == ord('s')


import numpy as np
# numpy or Numerical0 Python is this library provides you with an array data structure that holds some benefits over
# Python lists, such as: being more compact, faster access in reading and writing items, being
# more convenient and more efficient. an array is basically nothing but pointers.
# It’s a combination of a memory address, a data type, a shape and strides:
# The data pointer indicates the memory address of the first byte in the array,
# The data type or dtype pointer describes the kind of elements that are contained within the array,
# The shape indicates the shape of the array, and
# The strides are the number of bytes that should be skipped in memory to go to the next element.
# If your strides are (10,1), you need to proceed one byte to get to the next column and 10 bytes to locate the next row.
# To make a numpy array, you can just use the np.array() function.
# import numpy as np
# Make the array `my_array` : my_array = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)
#         numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
# Print `my_array`: print(my_array)
# Print out memory address: print(my_array.data)
# Print out the shape of `my_array`: print(my_array.shape)
# Print out the data type of `my_array`: print(my_array.dtype)
# Print out the stride of `my_array`: print(my_array.strides)
# np.zeros((2,3,4),dtype=np.int16) - 2 matrices, 3 x 2 each
# np.random.random((2,2))
# np.empty((3,2)) : numpy.empty(shape, dtype=float, order='C') - 3x3 zero matrix
# np.full((2,2),7) : numpy.full(shape, fill_value, dtype=None, order='C')
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Returns num evenly spaced samples, calculated over the interval [start, stop].
# x, y, z = np.loadtxt('data.txt',skiprows=1,unpack=True) you skip the first row and you return the columns as separate arrays with unpack=TRUE
# x = np.arange(0.0,5.0,1.0): np.savetxt('test.out', x, delimiter=',')
# np.add(x,y)

import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import numpy as np
# Prepare the data x = np.linspace(0, 10, 100)
# Plot the data plt.plot(x, x, label='linear')
# Add a legend plt.legend()
# Show the plot plt.show()
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
# ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
# ax.set_xlim(0.5, 4.5)
# plt.show()

import os
from random import shuffle
from tqdm import tqdm

import tensorflow as tf

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


TRAIN_DIR = 'training_data/train'
TEST_DIR = 'training_data/test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogsVScats-{}-{}.model'.format(LR, '6conv-basic')


def label_img(img):
	word_label = img.split('.')[-3]
	if word_label == 'cat': return [1, 0]
	elif word_label == 'dog': return [0, 1]


def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR, img)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		training_data.append([np.array(img), np.array(label)])
	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data


def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR, img)
		img_num = img.split('.')[0]
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img), img_num])
	np.save('test_data.npy', testing_data)
	return testing_data


def create_model():
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 32, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 64, 2, activation='relu')
	convnet = max_pool_2d(convnet, 2)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')
	return model


def trainModel(model, train_data):
	train = train_data[:-500]
	test = train_data[-500:]

	X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y = [i[1] for i in train]

	test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	test_y = [i[1] for i in test]

	model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
	model.save(MODEL_NAME)

def testModel(model, test_data):
	fig = plt.figure()
	for num, data in enumerate(test_data[:12]):
		# cat: [1, 0]
		# dog: [0, 1]

		img_num = data[1]
		img_data = data[0]

		y = fig.add_subplot(3,4,num+1)
		orig = img_data
		data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

		model_out = model.predict([data])[0]

		if np.argmax(model_out) == 1:
			str_label = 'Dog'
		else:
			str_label = 'Cat'

		y.imshow(orig, cmap='gray')
		plt.title(str_label)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)
	plt.show()

	# with open('submission-file.csv', 'w') as f:
	# 	f.write('id,label\n')

	# with open('submission-file.csv', 'a') as f:
	# 	for data in tqdm(test_data):
	# 		img_num = data[1]
	# 		img_data = data[0]
	# 		orig = img_data
	# 		data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	# 		model_out = model.predict([data])[0]
	# 		f.write('{},{}\n'.format(img_num, model_out[1]))

def main():
	if os.path.exists('train_data.npy'):
		train_data = np.load('train_data.npy')
		print('Train Data Loaded!!')
	else:
		train_data = create_train_data()

	if os.path.exists('test_data.npy'):
		test_data = np.load('test_data.npy')
		print('Test Data Loaded!!')
	else:
		test_data = process_test_data()

	model = create_model()
	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		model.load(MODEL_NAME)
		print('Model Loaded!')

	trainModel(model, train_data)

	testModel(model, test_data)

if __name__ == "__main__":
	main()	

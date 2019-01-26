# import the necessary packages
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
import cv2

print("Reading DataSets...")
trainData, trainLabels = read_hoda_dataset('./DigitDB/Train 60000.cdb')
validationData, validationLabels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb')
testData, testLabels = read_hoda_dataset('./DigitDB/Test 20000.cdb')


# handle matrix for when Keras is using "channels first" ordering (Theano).
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 32, 32))
	validationData = validationData.reshape((validationData.shape[0], 1, 32, 32))
	testData = testData.reshape((testData.shape[0], 1, 32, 32))

# handle matrix for when Keras is using "channels last" ordering (Tensorflow).
else:
	trainData = trainData.reshape((trainData.shape[0], 32, 32, 1))
	validationData = validationData.reshape((validationData.shape[0], 32, 32, 1))
	testData = testData.reshape((testData.shape[0], 32, 32, 1))


# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
validationData = validationData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0


# transform the training and testing labels into vectors in the range [0, classes]
# eg: 10 => [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]
trainLabels = np_utils.to_categorical(trainLabels, 10)
validationLabels = np_utils.to_categorical(validationLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)


# initialize the model
print("Compiling model...")
model = Sequential()
inputShape = (32, 32, 1)

# if Keras is using "channels first" (Theano), update the input shape
if K.image_data_format() == "channels_first":
	inputShape = (1, 32, 32)

# Convolution Layer 1
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
# SubSampling 1
model.add(AveragePooling2D())

# Convolution Layer 2
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# SubSampling 2
model.add(AveragePooling2D())

model.add(Flatten())

# Connected Layer 1
model.add(Dense(units=120, activation='relu'))

# Connected Layer 2
model.add(Dense(units=84, activation='relu'))

# Connected Layer 3
model.add(Dense(units=10, activation='softmax'))

# get an analysis on model params
model.summary()


# uncomment below line for using "pre-trained" model's weights.
# model.load_weights('weights.hdf5')


model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

print("Training...")
# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
						 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
						 horizontal_flip=True, fill_mode="nearest")

# train the network
H = model.fit_generator(aug.flow(trainData, trainLabels, batch_size=BS),
						validation_data=(validationData, validationLabels),
						steps_per_epoch=len(trainData) // BS,
						epochs=EPOCHS)

# show the accuracy on the testing set
print("Evaluating...")
(loss, accuracy) = model.evaluate(validationData, validationLabels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# save model's weights for using in future "pre-trained" models.
model.save_weights('weights.hdf5', overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# predict the digit's image label (classification)
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)

	# extract the image from the testData if Keras is using "channels_first" ordering (Theano).
	if K.image_data_format() == "channels_first":
		image = (testData[i][0] * 255).astype("uint8")

	# otherwise we are using "channels_last" ordering
	else:
		image = (testData[i] * 255).astype("uint8")

	# merge the channels into one image
	image = cv2.merge([image] * 3)

	# resize the image from a 32 x 32 image to a 256 x 256 image so we can better see it
	image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

	# threshold binary image for better view.
	lower_black = np.array([1, 1, 1], dtype="uint8")
	upper_black = np.array([255, 255, 255], dtype="uint8")
	image = cv2.inRange(image, lower_black, upper_black)

	# show the image and prediction
	cv2.putText(image, str(prediction[0]), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
	print("Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)


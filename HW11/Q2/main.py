# import the necessary packages
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
import time
import json

#
# exampleLayersDict = {
# 	"train_data": (trainData, trainLabels),
# 	"validation_data": (validationData, validationLabels),
# 	"input_shape": (32, 32, 1),
# 	"conv_layers": {
# 		1: {
# 			"filters_number": 6,
# 			"kernel_size": (3, 3),
# 			"activation_function": 'relu'
# 		},
# 		2: {
# 			"filters_number": 16,
# 			"kernel_size": (3, 3),
# 			"activation_function": 'relu'
# 		}
# 	},
# 	"connected_layers": {
# 		1: {
# 			"units": 120,
# 			"activation_function": 'relu'
# 		},
# 		2: {
# 			"units": 84,
# 			"activation_function": 'relu'
# 		},
# 		3: {
# 			"units": 10,
# 			"activation_function": 'softmax'
# 		}
# 	},
# 	"loss_method": 'categorical_crossentropy',
# 	"optimizer": Adam(), # or SGD(lr=0.1), ...
# 	"data_augmentation_method": ImageDataGenerator(rotation_range=20, zoom_range=0.15,
# 						 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
# 						 horizontal_flip=True, fill_mode="nearest"),
# 	"epochs": 100,
# 	"batch_size": 32,
# }
#


def model_generator(model_dict):
	start_time = time.time()

	# initialize the model
	print("Initialize model...")
	model = Sequential()

	# create convolutional layers
	print("Add Convolutional Layers...")
	convLayers = model_dict["conv_layers"]
	for convLayerNum in convLayers:
		if convLayerNum == 1:
			model.add(
				Conv2D(
					filters=convLayers[convLayerNum]["filters_number"],
					kernel_size=convLayers[convLayerNum]["kernel_size"],
					activation=convLayers[convLayerNum]["activation_function"],
					input_shape=model_dict["input_shape"]
				)
			)
		else:
			model.add(
				Conv2D(
					filters=convLayers[convLayerNum]["filters_number"],
					kernel_size=convLayers[convLayerNum]["kernel_size"],
					activation=convLayers[convLayerNum]["activation_function"],
				)
			)
		# add pooling layer just after each conv layer
		model.add(AveragePooling2D())

	# flatten last convolutional layer output to use in connected layers
	print("Add Flatten Layer...")
	model.add(Flatten())

	# create connected layers
	print("Add Connected Layers...")
	connectedLayers = model_dict["connected_layers"]
	for concLayerNum in connectedLayers:
		model.add(
			Dense(
				units=connectedLayers[concLayerNum]["units"],
				activation=connectedLayers[concLayerNum]["activation_function"]
			)
		)

	# show model's summary
	model.summary()
	model.summary(print_fn=lambda x: log.write(x + '\n'))

	# set model's hyper methods
	print("Compiling...")
	model.compile(
		loss=model_dict["loss_method"],
		optimizer=model_dict["optimizer"],
		metrics=['accuracy']
	)

	# train the model
	print("Training...")
	H = model.fit_generator(model_dict["data_augmentation_method"].flow(
		model_dict["train_data"][0], model_dict["train_data"][1], batch_size=model_dict["batch_size"]
	),
		validation_data=model_dict["validation_data"],
		steps_per_epoch=len(model_dict["train_data"][0]) // model_dict["batch_size"],
		epochs=model_dict["epochs"]
	)

	# model evaluating
	print("Evaluating...")
	(loss, accuracy) = model.evaluate(model_dict["validation_data"][0], model_dict["validation_data"][1])
	print("Accuracy: {:.2f}%".format(accuracy * 100))
	log.write("model accuracy: {:.2f}%\n".format(accuracy * 100))
	print("total time: {} seconds\n".format(time.time() - start_time))
	log.write("total time: {} seconds\n\n\n".format(time.time() - start_time))


if __name__ == '__main__':

	log = open("log.txt", "w")
	log.write('#######################START########################\n')

	print("Reading DataSets...")
	trainData, trainLabels = read_hoda_dataset('./DigitDB/Train 60000.cdb')
	validationData, validationLabels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb')

	# handle matrix for when Keras is using "channels first" ordering (Theano).
	# see this: https://stackoverflow.com/questions/39815518/keras-maxpooling2d-layer-gives-valueerror
	if K.image_data_format() == "channels_first":
		trainData = trainData.reshape((trainData.shape[0], 1, 32, 32))
		validationData = validationData.reshape((validationData.shape[0], 1, 32, 32))

	# handle matrix for when Keras is using "channels last" ordering (Tensorflow).
	else:
		trainData = trainData.reshape((trainData.shape[0], 32, 32, 1))
		validationData = validationData.reshape((validationData.shape[0], 32, 32, 1))

	# scale data to the range of [0, 1]
	trainData = trainData.astype("float32") / 255.0
	validationData = validationData.astype("float32") / 255.0

	# transform the training and testing labels into vectors in the range [0, classes]
	# eg: 10 => [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]
	trainLabels = np_utils.to_categorical(trainLabels, 10)
	validationLabels = np_utils.to_categorical(validationLabels, 10)

	models = [
		{
			"train_data": (trainData, trainLabels),
			"validation_data": (validationData, validationLabels),
			"input_shape": (1, 32, 32) if K.image_data_format() == "channels_first" else (32, 32, 1),
			"conv_layers": {
				1: {
					"filters_number": 3,
					"kernel_size": (3, 3),
					"activation_function": 'relu'
				},
				2: {
					"filters_number": 9,
					"kernel_size": (3, 3),
					"activation_function": 'relu'
				}
			},
			"connected_layers": {
				1: {
					"units": 120,
					"activation_function": 'relu'
				},
				2: {
					"units": 84,
					"activation_function": 'relu'
				},
				3: {
					"units": 10,
					"activation_function": 'softmax'
				}
			},
			"loss_method": 'categorical_crossentropy',
			"optimizer": Adam(),  # or SGD(lr=0.1), ...
			"data_augmentation_method": ImageDataGenerator(rotation_range=20, zoom_range=0.15,
														   width_shift_range=0.2, height_shift_range=0.2,
														   shear_range=0.15,
														   horizontal_flip=True, fill_mode="nearest"),
			"epochs": 10,
			"batch_size": 32,
		},
		{
			"train_data": (trainData, trainLabels),
			"validation_data": (validationData, validationLabels),
			"input_shape": (1, 32, 32) if K.image_data_format() == "channels_first" else (32, 32, 1),
			"conv_layers": {
				1: {
					"filters_number": 6,
					"kernel_size": (3, 3),
					"activation_function": 'relu'
				},
				2: {
					"filters_number": 16,
					"kernel_size": (3, 3),
					"activation_function": 'relu'
				}
			},
			"connected_layers": {
				1: {
					"units": 120,
					"activation_function": 'relu'
				},
				2: {
					"units": 84,
					"activation_function": 'relu'
				},
				3: {
					"units": 10,
					"activation_function": 'softmax'
				}
			},
			"loss_method": 'categorical_crossentropy',
			"optimizer": Adam(), # or SGD(lr=0.1), ...
			"data_augmentation_method": ImageDataGenerator(rotation_range=20, zoom_range=0.15,
				width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
				horizontal_flip=True, fill_mode="nearest"),
			"epochs": 10,
			"batch_size": 32,
		},
	]

	for model in models:
		print("Running Generator on: {}\n".format(json.dumps({
			"input_shape": model["input_shape"],
			"conv_layers": model["conv_layers"],
			"connected_layers": model["connected_layers"],
			"loss_method": model["loss_method"],
			"optimizer": model["optimizer"].__class__.__name__,
			"data_augmentation_method": model["data_augmentation_method"].__class__.__name__,
			"epochs": model["epochs"],
			"batch_size": model["batch_size"],
		}, sort_keys=False, indent=2)))
		log.write("Running Generator on: {}\n".format(json.dumps({
			"input_shape": model["input_shape"],
			"conv_layers": model["conv_layers"],
			"connected_layers": model["connected_layers"],
			"loss_method": model["loss_method"],
			"optimizer": model["optimizer"].__class__.__name__,
			"data_augmentation_method": model["data_augmentation_method"].__class__.__name__,
			"epochs": model["epochs"],
			"batch_size": model["batch_size"],
		}, sort_keys=False, indent=2)))
		model_generator(model)

	log.write('#######################END#########################\n')
	log.close()


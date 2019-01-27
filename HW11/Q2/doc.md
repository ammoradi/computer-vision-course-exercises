CNN Analysis
------------
in this question, we implement a **model generator** function located in `main.py`
which creates a convolutional neural network based on argument object(dictionary)
that will be run in a loop of different models arguments
to compare and analysis hyper parameters of a CNN.

Model Generator Arguments
-------------------------
there is one dictionary as argument containing hyper parameters
of CNN model for model generator function, structured like this:
```
{
    "train_data": (trainImages, trainLabels),
    "validation_data": (validationImages, validationLabels),
    "input_shape": (32, 32, 1),
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
}
```
as you see, the all keys of dictionary
is a dynamic hyper parameter for CNN model which can be changed.

Parameters for Analysis
------------------
For each model hyper parameters, the result have these parameters that will be used for analysis:
* Number of network parameters
* Accuracy
* Execution time

For each execution of `main.py`, the `log.txt` file will be filled with
needed information for analysis. (there is some examples available in `Logs/` directory)

Influence of HyperParameters
-------------------
Totally, there is not constant rule for a network that if you increase/decrease
a parameter from 0 to infinity/infinity to 0 the accuracy of model will be increased up to 100%.

it is because each network's underfitting/overfitting range is different and it is highly depends on
data sets and network architecture.
for example clipping, rotating and many other data augmentations is good for
preventing a model to be overfitted by increasing Variety of input images.
but if you increase images variety more than enough, your model will be underfitted!
the other example is number of filters and convolutional layers,
increase the number of these parameters may remove the essential features
of data set images!

with all this, if we suppose change of hyper parameters is in range of well fitted model,
we can consider these rules for these parameters:

* **epochs**: Number of epochs is the number of times the whole training data is shown to the network while training.
more epochs increases accuracy and time of execution. it has no effect on network parameters.

* **batch size**: is the number of sub samples given to the network after which parameter update happens.
A good default for batch size might be 32 (small values).

* **number of connected layers**: It is usually good to add more layers until the test error no longer improves. The trade off is that it is computationally expensive to train the network. Having a small amount of units may lead to underfitting while having more units are usually not harmful with appropriate regularization.

References
----------
* [https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a](https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a)
* [https://towardsdatascience.com/a-walkthrough-of-convolutional-neural-network-7f474f91d7bd](https://towardsdatascience.com/a-walkthrough-of-convolutional-neural-network-7f474f91d7bd)
* [https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters](https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters)

OCR of Hand-written Data using kNN
====================
**course tags** : `machine learning`, `computer vision`

**algorithm** : [k-Nearest Neighbour (kNN)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html#knn-understanding)

**libraries** : [OpenCV](https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html), [HodaDatasetReader](https://github.com/amir-saniyan/HodaDatasetReader)

**input** : data set of handwritten farsi digits

**output** : labels of predicted numbers and data accuracy

Project Structure
--------------------
* _/DigitDB_: contains [these files](https://github.com/amir-saniyan/HodaDatasetReader/tree/master/DigitDB) as train, test, remained dataset (fill this directory if it is empty).
* _HodaDatasetReader.py_: To read Hoda `.cdb` files as images
* _main.py_ : where the project code implemented. **run this file**
* _log.txt_ : a text file which be filled automatically by `main.py`. it used as a store and history for all printed lines of `main.py` to make conclusion

Main.py step-by-step description
--------------------
1. read data sets files ( each dataset contains images as a numpy 2D array and labels as a numpy 1D array).
2. run `knn.train()` with train dataset to train numbers of images.
3. run `knn.findNearest()` with test dataset to predict numbers of images.
4. calculate accuracy by [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) with average precision
5. append test data set to train data set for training and use remaining data set for testing. (the old test data set used for **validation** step)
6. goto 2

**NOTE:** the feature area of search is flatten array of image pixels (simplest feature set)

Accuracy Calculating
--------------------
for each number of 0 to 9, on each index of prediction/result the program track true positive, true negative, false positive and false negative states.
then calculates the sum of each of four above states for all 0 to 9 numbers.
for example false positive state of number 2 happened when on each number prediction, actual number were not 2 and program detected 2.
the program will calculate accuracy by this formula:

`accuracy = (true positive + true negative) / (test_size * 10) * 100`

the `* 10` used for test size because the program calculates **sum** of true positives (and three other states) of **10** numbers.

Conclusion
-------------------
* the kNN algorithm is slow, because it iterates all elements.
* the accuracy rate of algorithm does not grow by increasing search size presently.
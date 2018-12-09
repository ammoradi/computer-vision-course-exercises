# *-* coding: utf-8 *-*

import cv2 as cv
import numpy as np
import time
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset


def train(knn, train_i, train_l, test_i, test_l, search_size):  # i = images, l = labels

    print("Training: END ({} seconds)".format(time.time() - start_time))
    log.write("Training: END ({} seconds)\n".format(time.time() - start_time))
    test_start_time = time.time()
    print("\nTesting: START...")
    log.write("\nTesting: START...\n")

    knn.train(train_i, cv.ml.ROW_SAMPLE, train_l)
    ret, results, neighbours, dist = knn.findNearest(test_i, search_size)

    print("Testing: END ({} seconds)".format(time.time() - test_start_time))
    log.write("Testing: END ({} seconds)\n".format(time.time() - test_start_time))

    results = np.reshape(results, -1)

    # uncomment these lines below if you want to see neighbours and distances matrix.
    # print("Neighbours:  \n{}\n".format(neighbours))
    # print("Distances:  \n{}\n".format(dist))
    print()
    print("Actual labels: ", test_l)
    log.write("Actual labels: {}\n".format(test_l))
    print("Predicted labels: ", results)
    log.write("Predicted labels: {}\n".format(results))
    print()

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(10):
        for j in range(len(results)):
            if test_l[j] == i and results[j] == i:
                tp += 1
            elif test_l[j] != i and results[j] != i:
                tn += 1
            elif test_l[j] == i and results[j] != i:
                fn += 1
            elif test_l[j] != i and results[j] == i:
                fp += 1

    print("False positives: ", fp)
    log.write("False positives: {}\n".format(fp))
    print("False negatives: ", fn)
    log.write("False negatives: {}\n".format(fn))
    print("True positives: ", tp)
    log.write("True positives: {}\n".format(tp))
    print("True negatives: ", tn)
    log.write("True negatives: {}\n".format(tn))

    accuracy = ((tp + tn) / (len(results) * 10)) * 100
    print("\nAccuracy: (true positives + true negatives) / (test size) * 100 = {}%".format(accuracy))
    log.write("\nAccuracy: (true positives + true negatives) / (test size) * 100 = {}% \n".format(accuracy))


log = open("log.txt", "w")
log.write('#######################START########################\n')

print('Reading train dataset (Train 60000.cdb)...')
log.write('Reading train dataset (Train 60000.cdb)...\n')
train_images, train_labels = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                               images_height=32,
                                               images_width=32,
                                               one_hot=False,
                                               reshape=True)


print('Reading test dataset (Test 20000.cdb)...')
log.write('Reading test dataset (Test 20000.cdb)...\n')
test_images, test_labels = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                             images_height=32,
                                             images_width=32,
                                             one_hot=False,
                                             reshape=True)


print('Reading remaining samples dataset (RemainingSamples.cdb)...')
log.write('Reading remaining samples dataset (RemainingSamples.cdb)...\n')
remaining_images, remaining_labels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb',
                                                       images_height=32,
                                                       images_width=32,
                                                       one_hot=False,
                                                       reshape=True)

opencv_knn = cv.ml.KNearest_create()

print("\nTraining: START...")
log.write("\nTraining: START...\n")
start_time = time.time()
train(opencv_knn, train_images, train_labels, test_images, test_labels, 3)

print("\nappend Test Data to Train Data for increasing accuracy.")
print("Remaining Dataset will be used for new Test.")
log.write("\nappend Test Data to Train Data for increasing accuracy.\n")
log.write("Remaining Dataset will be used for new Test.\n")

start_time = time.time()
new_train_images = np.concatenate((train_images, test_images), axis=0)
new_train_labels = np.concatenate((train_labels, test_labels), axis=0)

print("\nTraining: START...")
log.write("\nTraining: START...\n")
train(opencv_knn, new_train_images, new_train_labels, remaining_images, remaining_labels, 5)
log.write('########################END#########################\n')
log.close()

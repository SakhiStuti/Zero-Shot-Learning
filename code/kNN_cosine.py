#########################################  
# kNN: k Nearest Neighbors  
  
# Input:      newInput: vector to compare to existing dataset (1xN)  
#             dataSet:  size m data set of known vectors (NxM)  
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   
              
# Output:     the most popular class label  
#########################################  
  
from numpy import *  
import operator  
from torch.nn import functional as F
import torch

# create a dataset which contains 4 samples with 2 classes  
def createDataSet():  
    # create a matrix: each row as a sample  
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])  
    labels = ['A', 'A', 'B', 'B'] # four samples and two classes  
    return group, labels  


# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row  
  
    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    newInput_rep = newInput.repeat(numSamples,1)
    #print(newInput_rep.shape)
    distance = 1 - F.cosine_similarity(newInput_rep, dataSet, dim=1)
    #print(distance.shape, distance)

    #distance-> (50,1)
    #print(distance, distance.shape)
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a descending order  
    sorted_dist, sortedDistIndices = torch.sort(distance,descending=False)#
    #print('Sorted_dist: {}'.format(sorted_dist[0]))
    sortedDistIndices = sortedDistIndices.numpy()
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
  
    classCount = {} # define a dictionary (can be append element)  
    for i in range(k):  
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]  
  
        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex
    #return sortedDistIndices   

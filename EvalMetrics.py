"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

"""
import numpy as np
def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8 )
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array). 
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels)) 

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)

def ErrorRateAt_N_Recall(recallpercnt, labels, scores):
    distances = 1.0 / (scores + 1e-8 )
    recall_point = recallpercnt
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)

## Error Rate      = FP     / (FP + TN)
## Precision Score = TP     / (TP + FP)
## Recall    Score = TP     / (FN + TP)
## Accuracy  Score = (TP+TN)/ (TP + FN + TN + FP)
def PrecisionAt_N_Recall(recallpercnt, labels, scores):
    distances = 1.0 / (scores + 1e-8 )
    recall_point = recallpercnt
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TP = np.sum(labels[:threshold_index] == 1)
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    FN = np.sum(labels[threshold_index:] == 1)
    precision_at_N_recall = float(TP)/float(TP+FP)
    return precision_at_N_recall

def one_class_recall(pred, lab):
    shape = pred.shape
    p_flat = pred.view(shape[0], -1)
    l_flat = lab.view(shape[0], -1)
    true_positive = (p_flat * l_flat).sum()
    negative = 1 - p_flat
    false_negative = (negative * l_flat).sum()
    return true_positive / (true_positive + false_negative)

def one_class_precision(pred, lab):
    shape = pred.shape
    p_flat = pred.view(shape[0], -1)
    l_flat = lab.view(shape[0], -1)
    true_positive = (p_flat * l_flat).sum()
    return true_positive / p_flat.sum()

def one_class_dice(pred, lab):
    shape = pred.shape
    p_flat = pred.view(shape[0], -1)
    l_flat = lab.view(shape[0], -1)
    true_positive = (p_flat * l_flat).sum()
    return (2. * true_positive) / (p_flat.sum() + l_flat.sum())



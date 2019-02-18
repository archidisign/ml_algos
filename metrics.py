def confusion_matrix(target, prediction):
    TP = FP = FN = TN = 0
    for i in range(len(target)):
        if prediction[i] > 0.5 and target[i] == 1:
            TP += 1
        elif prediction[i] > 0.5 and target[i] == 0:
            FP += 1
        elif target[i] == 1:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN

def metrics(TP, FP, FN, TN, verbose):
    # Accuracy
    accuracy = (TP+TN)/(TP + FP + FN + TN)
    # Precision
    precision = TP/(TP+FP)
    # Recall
    recall = TP/(TP+FN)
    # F1-Measure
    f1 = 2*precision*recall/(precision+recall)
    if verbose:
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 measure: " + str(f1))
    return accuracy, precision, recall, f1
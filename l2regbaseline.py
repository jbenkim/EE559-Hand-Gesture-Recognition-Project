#l2 regularized baseline
#kaggle api: KGAT_3df18712301b85228c4c70d49930d020
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image #open/resize images

dataset = "leapGestRecogMIXED"
imagesize = (64, 64) #64 x 64 for faster training
validationsplit = 0.15
testsplit = 0.15
np.random.seed(42)
learningrates = [0.1, 0.05, 0.01]
batch = [64]
lambdas = [1e-4, 1e-3, 1e-2] #l2 regularization strengths
iterations = 20

def softmax(z): #row logits to probs
    z = z - np.max(z, axis = 1, keepdims = True) #numerical stability of subtracting max logit from each row
    exponential = np.exp(z)
    return exponential / np.sum(exponential, axis = 1, keepdims = True)

def predict(x, weight): #prediction function
    classprobabilities = softmax(x @ weight) #linear model
    return np.argmax(classprobabilities, axis = 1)

def crossentropyloss(x, yonehot, weight): #ce loss function
    classprobabilities = softmax(x @ weight)
    loss = -np.mean(np.sum(yonehot * np.log(classprobabilities + 1e-12), axis = 1)) #cross entropy loss
    return loss

def accuracy(ytrue, yprediction): #accuracy metrics
    return np.mean(ytrue == yprediction)

def macrof1(ytrue, yprediction, numclasses):
    f1score = []
    for i in range(numclasses):
        truepositives = np.sum((ytrue == i) & (yprediction == i))
        falsepositives = np.sum((ytrue != i) & (yprediction == i))
        falsenegatives = np.sum((ytrue == i) & (yprediction != i))
        precision = truepositives / (truepositives + falsepositives + 1e-12) #1e-12 to prevent dividing by 0
        recall = truepositives / (truepositives + falsenegatives + 1e-12)
        if precision + recall == 0:
            f1score.append(0.0)
        else:
            f1score.append(2 * precision * recall / (precision + recall)) #2pr/p+r
    return np.mean(f1score) #average across classes for macro f1

def confusionmatrix(ytrue, yprediction, numclasses):
    matrix = np.zeros((numclasses, numclasses), dtype = int) #rows true class, columns predicted
    for i in range(len(ytrue)): #count predictions
        matrix[ytrue[i], yprediction[i]] += 1
    return matrix

def onehot(y, numclasses): #one hot encoding for ce loss
    yonehot = np.zeros((len(y), numclasses), dtype = np.float32)
    yonehot[np.arange(len(y)), y] = 1.0 #select row index -> select class column -> set to 1.0
    return yonehot

def l2loss(weight, lambdanum): #computes l2 penalty
    regmask = np.ones_like(weight, dtype = np.float32) #matrix of ones with same shape as weight
    regmask[0, :] = 0.0 #set first row of mask to 0
    return 0.5 * lambdanum * np.sum((weight * regmask) ** 2) #l2 penalty

def totalloss(x, yonehot, weight, lambdanum): #full loss w l2 reg
    return crossentropyloss(x, yonehot, weight) + l2loss(weight, lambdanum) #add ce loss and l2 penalty

#load data
xflattenedimages = []
yintegerlabels = []
gestures = set() #gets all unique gesture folder names

#find all gesture class names
for person in os.listdir(dataset): #loop through everything inside each person folder
    personpath = os.path.join(dataset, person) #create path to person folder
    if not os.path.isdir(personpath): #check if it's actually a folder
        continue
    #loop through everything inside each gesture folder and add gesture name to set of unique gestures
    for gesture in os.listdir(personpath):
        gesturepath = os.path.join(personpath, gesture)
        if os.path.isdir(gesturepath):
            gestures.add(gesture) #add gesture name to unique set

#sort class names and assign each gesture an integer index: "01_palm" to 0, "02_l" to 1
gestures = sorted(list(gestures))
classtoindex = {g: i for i, g in enumerate(gestures)}
numclasses = len(gestures)

#load dataset into numpy arrays
for person in os.listdir(dataset):
    personpath = os.path.join(dataset, person)
    if not os.path.isdir(personpath):
        continue
    for gesture in os.listdir(personpath):
        gesturepath = os.path.join(personpath, gesture)
        if not os.path.isdir(gesturepath):
            continue
        label = classtoindex[gesture] #convert gesture name into integer label
        #loop through files inside gesture folder
        for file in os.listdir(gesturepath):
            if not file.lower().endswith((".png", ".jpg")):
                continue
            imagepath = os.path.join(gesturepath, file)
            image = Image.open(imagepath).convert("RGB") #open image and convert to RGB
            image = image.resize(imagesize) #resize to 64 x 64
            image = np.array(image, dtype = np.float32) / 255.0
            xflattenedimages.append(image.flatten()) #imagine becomes a vector of length 64 x 64 x 3 = 12288
            yintegerlabels.append(label) #store integer label for this image
X = np.array(xflattenedimages, dtype = np.float32) #list of vectors representing images
y = np.array(yintegerlabels, dtype = np.int64) #list of labels corresponding to each image
classcounts = np.bincount(y, minlength=numclasses)

#class distribution plot
plt.figure(figsize = (8, 5)) 
plt.bar(gestures, classcounts)
plt.title("class distribution")
plt.xlabel("gesture class")
plt.ylabel("images per class")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("baselineclassdistribution.png")
plt.show()

#check
print("classes:", gestures)
print("total samples:", len(X))

#shuffle/split
randomindices = np.random.permutation(len(X)) #creates random ordering of indices to shuffle data (prevents bias)
#reorder X and y 
X = X[randomindices]
y = y[randomindices]
testsamples = int(len(X) * testsplit)
validationsamples = int(len(X) * validationsplit)
#first chunk of shuffled data
xtest = X[:testsamples]
ytest = y[:testsamples]
#second chunk for val
xvalidation = X[testsamples:testsamples + validationsamples]
yvalidation = y[testsamples:testsamples + validationsamples]
#remiaining chunk for training
xtrain = X[testsamples + validationsamples:]
ytrain = y[testsamples + validationsamples:]
print("training:", len(xtrain))
print("validation:", len(xvalidation))
print("test:", len(xtest))

#bias term (lr model)
xtrain = np.column_stack((np.ones(xtrain.shape[0]), xtrain))
xvalidation = np.column_stack((np.ones(xvalidation.shape[0]), xvalidation))
xtest = np.column_stack((np.ones(xtest.shape[0]), xtest))

#trianing labels to one hot format
ytrainonehot = onehot(ytrain, numclasses)
yvalonehot = onehot(yvalidation, numclasses)
ytestonehot = onehot(ytest, numclasses)

#l2 training
def training(learningrate, batchsize, lambdanum):
    weight = np.zeros((xtrain.shape[1], numclasses), dtype = np.float32) #initialize weights to 0s
    regmask = np.ones_like(weight, dtype = np.float32)
    regmask[0, :] = 0.0 #prevents bias row from being regularized
    #learning curves
    trainlosshistory = []
    validationlosshistory = []
    trainaccuracyhistory = []
    validationaccuracyhistory = []
    validationf1history = []

    #mini batch gradient descent training loop
    for epoch in range(iterations):
        rand = np.random.permutation(len(xtrain)) #creates random ordering of training sample indices (i dont want to always train on same order of samples each epoch)
        #shuffle training data/labels
        xtrainshuffled = xtrain[rand]
        ytrainonehotshuffled = ytrainonehot[rand]
        #mini-batch loop
        for start in range(0, len(xtrainshuffled), batchsize):
            endindex = min(start + batchsize, len(xtrainshuffled)) #find ending index of current batch
            #get current batch of images/labels
            xbatch = xtrainshuffled[start:endindex]
            ybatch = ytrainonehotshuffled[start:endindex]
            classprobabilities = softmax(xbatch @ weight) #Y = softmax(XW)
            gradient = (xbatch.T @ (classprobabilities - ybatch)) / len(xbatch) #gradient of softmax ce wrt weights
            gradient += lambdanum * weight * regmask #add l2 gradient to softmax ce gradient
            weight = weight - learningrate * gradient
        trainloss = totalloss(xtrain, ytrainonehot, weight, lambdanum)
        validationloss = totalloss(xvalidation, yvalonehot, weight, lambdanum) #validation loss of ce and l2 penalty
         #prediction labels for training/validation
        ytrainpredictions = predict(xtrain, weight)
        yvalidationpredictions = predict(xvalidation, weight)
        #training/validation accuracy/macrof1
        trainingaccuracy = accuracy(ytrain, ytrainpredictions)
        validationaccuracy = accuracy(yvalidation, yvalidationpredictions)
        validationf1 = macrof1(yvalidation, yvalidationpredictions, numclasses)
        #store metrics
        trainlosshistory.append(trainloss)
        validationlosshistory.append(validationloss)
        trainaccuracyhistory.append(trainingaccuracy)
        validationaccuracyhistory.append(validationaccuracy)
        validationf1history.append(validationf1)
    history = {"trainloss": trainlosshistory, "validationloss": validationlosshistory, "trainingaccuracy": trainaccuracyhistory, "validationaccuracy": validationaccuracyhistory, "validationf1": validationf1history}
    return weight, history

#hyperparameter tuning
hyperparameterresults = []
bestvalidationf1 = -1 #start with -1 so any real f1 is better
bestweight = None
besthistory = None
bestparameters = None
print("hyperparameter tuning results: ")

for i in learningrates:
    for j in batch:
        for lambdanum in lambdas: #loop through l2 strengths
            weight, history = training(i, j, lambdanum) #training with one learning rate and batch size
            finalvalidationaccuracy = history["validationaccuracy"][-1] #gets final validation accuracy from last epoch
            finalvalidationf1 = history["validationf1"][-1]
            hyperparameterresults.append((i, j, lambdanum, finalvalidationaccuracy, finalvalidationf1))
            print(f"learning rate: {i}. batch size: {j}. lambda: {lambdanum}. validation accuracy: {finalvalidationaccuracy:.4f}. validation macrof1: {finalvalidationf1:.4f}")
            #check if model is best so far -> make updates
            if finalvalidationf1 > bestvalidationf1:
                bestvalidationf1 = finalvalidationf1
                bestweight = weight
                besthistory = history
                bestparameters = (i, j, lambdanum)
print("best hyperparameters")
print("best learning rate:", bestparameters[0])
print("batch size:", bestparameters[1])

#final evaluation
ytestpredictions = predict(xtest, bestweight)
testloss = totalloss(xtest, ytestonehot, bestweight, bestparameters[2])
testaccuracy = accuracy(ytest, ytestpredictions)
testf1 = macrof1(ytest, ytestpredictions, numclasses)
testconfusionmatrix = confusionmatrix(ytest, ytestpredictions, numclasses)
print("final test metrics:")
print(f"test loss: {testloss:.4f}")
print(f"test accuracy: {testaccuracy:.4f}")
print(f"test macrof1: {testf1:.4f}")
print("test confusion matrix:")
print(testconfusionmatrix)

#hyperparameter summary
print("hyperparameter summary")
print("learning rate - batch size - lambda - validation accuracy - validation macrof1")
for i, j, lambdanum, validationaccuracy, validationf1 in hyperparameterresults:
    print(f"{i:<13} - {j:<10} - {lambdanum:<10} - {validationaccuracy:.4f} - {validationf1:.4f}")
print("lambda:", bestparameters[2])

#plot
epochs = range(1, iterations + 1)
#training vs validation loss
plt.figure()
plt.plot(epochs, besthistory["trainloss"], label = "train Loss")
plt.plot(epochs, besthistory["validationloss"], label = "validation Loss")
plt.title("l2 regularized baseline: training vs validation loss")
plt.xlabel("epochs")
plt.ylabel("cross entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("l2regbaselinelosscurve.png")
plt.show()
#training vs validation accuracy
plt.figure()
plt.plot(epochs, besthistory["trainingaccuracy"], label = "train accuracy")
plt.plot(epochs, besthistory["validationaccuracy"], label = "validation accuracy")
plt.title("l2 regularized baseline: training vs validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("l2regbaselineaccuracycurve.png")
plt.show()
#validation macrof1
plt.figure()
plt.plot(epochs, besthistory["validationf1"], label = "validation macrof1")
plt.title("l2 regularized baseline: validation macro F1")
plt.xlabel("epochs")
plt.ylabel("macrof1")
plt.legend()
plt.tight_layout()
plt.savefig("l2regbaselinemacrof1curve.png")
plt.show()

#test confusion matrix plot
plt.figure(figsize = (8, 6))
plt.imshow(testconfusionmatrix, cmap = "Blues")
plt.title("l2 regularized baseline: test confusion matrix")
plt.xlabel("predicted class")
plt.ylabel("true class")
plt.xticks(np.arange(numclasses), gestures, rotation = 45)
plt.yticks(np.arange(numclasses), gestures)
for i in range(numclasses): #write count in each cell
    for j in range(numclasses):
        plt.text(j, i, testconfusionmatrix[i, j], ha = "center", va = "center")
plt.colorbar()
plt.tight_layout()
plt.savefig("l2regbaselinetestconfusinonmatrix.png")
plt.show()
#save best model
np.savez("l2regbaselinebestmodel.npz", weight = bestweight, labels = np.array(gestures), learningrate = bestparameters[0], batch = bestparameters[1], lambdanum = bestparameters[2], testaccuracy = testaccuracy, testmacrof1 = testf1)
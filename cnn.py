#cnn fine tuning on just 7 classes
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn #pyorch tensors, cnn model, loss functions, etc.
from torch.utils.data import Dataset, DataLoader, random_split #for custom dataset and data loading
from torchvision import transforms #resize images and convert into tensors

dataset = "leapGestRecog"
mydataset = "myhandgestures"
specificgestures = ["01_palm", "02_l", "03_fist", "05_thumb", "07_ok", "08_palm_moved", "09_c"]
imagesize = 64
batch = 64
epochs = 10
learningrate = 0.001
validationsplit = 0.15
testsplit = 0.15
#random split reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HandGestureDataset(Dataset): #custom pytorch dataset to load my gesture images
    def __init__(self, dataset, transform = None, classidx = None, specificgestures = None): #initializes dataset path, transform, samples, class labels, and gestures
        self.dataset_path = dataset #stores dataset folder path
        self.transform = transform #stores transofmr to apply to images
        self.samples = [] #stores image paths/labels
        self.gestures = set()
        self.specificgestures = specificgestures #stores selected gesture classes
        #handle kaggle folder (leapGestRecog/person/gesture/images)
        for person in os.listdir(dataset):  #loop through everything inside each person folder
            personpath = os.path.join(dataset, person ) #create path to person folder
            if not os.path.isdir(personpath): #check if it's actually a folder
                continue
            #loop through everything inside each gesture folder and add gesture name to set of unique gestures
            for gesture in os.listdir(personpath):
                gesturepath = os.path.join(personpath, gesture) #creates full path to gesture folder
                if os.path.isdir(gesturepath):
                    if self.specificgestures is None or gesture in self.specificgestures: #if no specific gestures were selected, keep all gestures. if specific gestures were selected, only keep this gesture if its in the list
                        self.gestures.add(gesture) #add gesture name to unique set

        #my personal dataset
        if len(self.gestures) == 0: #runs if no gesture classes were found from kaggle folder
            for gesture in os.listdir(dataset):
                gesturepath = os.path.join(dataset, gesture)
                if os.path.isdir(gesturepath):
                    if self.specificgestures is None or gesture in self.specificgestures:
                        self.gestures.add(gesture)
        self.gestures = sorted(list(self.gestures)) #converts set into sorted list

        if classidx is None: #check if class mapping is there
            self.classidx = {g: i for i, g in enumerate(self.gestures)} #create integer labels for each gesture
        else:
            self.classidx = classidx #use an existing class mapping
        #load images from kaggle folder
        for person in os.listdir(dataset):
            personpath = os.path.join(dataset, person)
            if not os.path.isdir(personpath):
                continue
            for gesture in os.listdir(personpath):
                gesturepath = os.path.join(personpath, gesture)
                if os.path.isdir(gesturepath) and gesture in self.classidx:
                    label = self.classidx[gesture]
                    for file in os.listdir(gesturepath):
                        if file.lower().endswith((".png", ".jpg")):
                            imagepath = os.path.join(gesturepath, file)
                            self.samples.append((imagepath, label))
        #load images from my dataset
        if len(self.samples) == 0:
            for gesture in os.listdir(dataset):
                gesturepath = os.path.join(dataset, gesture)
                if not os.path.isdir(gesturepath):
                    continue
                if gesture not in self.classidx:
                    continue
                label = self.classidx[gesture]
                for file in os.listdir(gesturepath):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        imagepath = os.path.join(gesturepath, file)
                        self.samples.append((imagepath, label))
    def __len__(self): #returns number of samples in dataset
        return len(self.samples)
    def __getitem__(self, index): #load one sample at position index
        imagepath, label = self.samples[index]
        image = Image.open(imagepath).convert("RGB") #opens image and converts to RGB
        if self.transform: #check if preprocessing exists
            image = self.transform(image) #apply transformations
        return image, label

#transforms
transform = transforms.Compose([transforms.Resize((imagesize, imagesize)), transforms.ToTensor()]) #resizes to 64 x 64, converts image from PIl to pytorch tensor, and normalizes pixel values to [0, 1]
fulldataset = HandGestureDataset(dataset, transform = transform, specificgestures = specificgestures) #creates datasetusing folder paths
print("classes: ", fulldataset.gestures)
print("total kaggle images: ", len(fulldataset))
numclasses = len(fulldataset.gestures) #number of gesture classes

#split
testsize = int(len(fulldataset) * testsplit)
validationsize = int(len(fulldataset) * validationsplit)
trainsize = len(fulldataset) - validationsize - testsize
traindataset, validationdataset, testdataset = random_split(fulldataset,[trainsize, validationsize, testsize], generator = torch.Generator().manual_seed(42)) #random split of dataset with reproducibility
#training/validation/test loaders
trainloader = DataLoader(traindataset, batch_size = batch, shuffle = True)
validationloader = DataLoader(validationdataset, batch_size = batch, shuffle = False)
testloader = DataLoader(testdataset, batch_size = batch, shuffle = False)
print("training: ", trainsize)
print("validation: ", validationsize)
print("test: ", testsize)
#load my own dataset
personaldataset = HandGestureDataset(mydataset, transform = transform, classidx = fulldataset.classidx, specificgestures = specificgestures) #dataset object of my personal hand gesture images, uses same class mapping and transformations as kaggle dataset
personalloader = DataLoader(personaldataset, batch_size = batch, shuffle = False) #data loader for my dataset
print("personal test samples: ", len(personaldataset))

#class distribution plot
labels = [label for _, label in fulldataset.samples]
classcounts = np.bincount(labels, minlength = numclasses) #counts number of images that belong to each class (minlength ensures we get a count for every class even if some classes have 0 images)
plt.figure(figsize = (8, 5))
plt.bar(fulldataset.gestures, classcounts)
plt.title("cnn dataset class distribution")
plt.xlabel("gesture class")
plt.ylabel("images per class")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("cnnclassdistribution.png")
plt.show()

#cnn model
class HandGestureCNN(nn.Module):
    def __init__(self, numclasses): #initializes cnn layers based on numclasses
        super(HandGestureCNN, self).__init__() #initalizes parent pytorch class
        self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), #first conv layer: input channels = 3 (RGB), output channels = 16, kernel size = 3x3, padding = 1 to preserve spatial dimensions with relu and 2x2 max pooling to reduce spatial dimensions by half
                                      nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), #second conv layer: input channels = 16, output channels = 32, kernel size = 3x3, padding = 1 with relu and 2x2 max pooling to reduce spatial dimensions by half again
                                      nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)) #third conv layer: input channels = 32, output channels = 64, kernel size = 3x3, padding = 1 with relu and 2x2 max pooling to reduce spatial dimensions by half again
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, numclasses)) #groups fully connected classification layers: flattens output of conv layers into vector, fully connected layer with 128 hidden units and relu, final fully connected layer that outputs numclasses logits for classification
    def forward(self, x): #how images move through cnn
        x = self.features(x) #passes image through convolution/pooling layers
        x = self.classifier(x) #passes output through fully connected layers
        return x #logits
model = HandGestureCNN(numclasses).to(device) #creates cnn
criterion = nn.CrossEntropyLoss() #cross entroyp loss
optimizer = torch.optim.Adam(model.parameters(), lr = learningrate) #adam optimizer 

def macrof1(ytrue, yprediction, numclasses):
    f1score = []
    ytrue = np.array(ytrue)
    yprediction = np.array(yprediction)
    for i in numclasses:
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

def evaluate(model, loader): #evalaute model performance on validaiton/test
    model.eval() #switches model to evaluation mode
    totalloss = 0.0
    correctpredictions = 0
    totalsamples = 0
    #f1 score
    truelabels = []
    predictionlabels = []

    with torch.no_grad(): #turns off gradient tracking for efficiency, less memory, and no training
        for images, labels in loader: #loops through batches from dataloaders
            #move data to cpu/gpu
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) #forward pass through cnn
            loss = criterion(outputs, labels) #ce loss
            totalloss += loss.item() #batch loss
            predictions = torch.argmax(outputs, dim = 1) #predicted class (highest score)
            correctpredictions += (predictions == labels).sum().item() #number of correct predictions
            totalsamples += labels.size(0) #number of samples
            #store results for f1 score
            truelabels.extend(labels.cpu().numpy())
            predictionlabels.extend(predictions.cpu().numpy())

    averageloss = totalloss / len(loader) 
    accuracy = correctpredictions / totalsamples
    uniqueclasses = sorted(list(set(truelabels))) #unique classes in dataset
    macrof1score = macrof1(truelabels, predictionlabels, uniqueclasses)
    return averageloss, accuracy, macrof1score, truelabels, predictionlabels

#train
trainlosshistory = []
validationlosshistory = []
trainaccuracyhistory = []
validationaccuracyhistory = []
validationf1history = []
for epoch in range(epochs):
    model.train() #switch to training mode
    #tracking variables
    runningloss = 0.0
    correctpredictions = 0
    totalsamples = 0

    for images, labels in trainloader: #loop through training batches
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad() #clears old gradients from previous batch
        loss.backward() #compute gradients using backprop
        optimizer.step() #update cnn weights
        runningloss += loss.item() #add batch loss to epoch loss total
        predictions = torch.argmax(outputs, dim = 1)
        correctpredictions += (predictions == labels).sum().item()
        totalsamples += labels.size(0)
    trainloss = runningloss / len(trainloader) #average training loss for the epoch
    trainingaccuracy = correctpredictions / totalsamples

    validationloss, validationaccuracy, validationf1, yvalidationtrue, yvalidationprediction = evaluate(model, validationloader) #evaluate model on validation data 
    #store metrics for plotting
    trainlosshistory.append(trainloss)
    validationlosshistory.append(validationloss)
    trainaccuracyhistory.append(trainingaccuracy)
    validationaccuracyhistory.append(validationaccuracy)
    validationf1history.append(validationf1)

    print(f"epoch {epoch + 1:02d}/{epochs}. "f"training loss: {trainloss:.4f}. " f"validation loss: {validationloss:.4f}. " f"training accuracy: {trainingaccuracy:.4f}. " f"validation accuracy: {validationaccuracy:.4f}. "f"validation macrof1: {validationf1:.4f}") #training progress

#final evaluation
testloss, testaccuracy, testf1, ytesttrue, ytestpredictions = evaluate(model, testloader) #evaluate trained cnn on kaggle test set
testconfusionmatrix = confusionmatrix(ytesttrue, ytestpredictions, numclasses)
print("final kaggle test metrics:")
print(f"test loss: {testloss:.4f}")
print(f"test accuracy: {testaccuracy:.4f}")
print(f"test macrof1: {testf1:.4f}")
print("kaggle test confusion matrix:")
print(testconfusionmatrix)

#evalaute kaggle trained cnn onto my images
personalloss, personalaccuracy, personalf1, ypersonaltrue, ypersonalprediction = evaluate(model, personalloader)
personalconfusionmatrix = confusionmatrix(ypersonaltrue, ypersonalprediction, numclasses)
print("final personal dataset metrics:")
print(f"personal loss: {personalloss:.4f}")
print(f"personal accuracy: {personalaccuracy:.4f}")
print(f"personal macrof1: {personalf1:.4f}")
print("personal dataset confusion matrix:")
print(personalconfusionmatrix)
torch.save({"learnedweights": model.state_dict(), "labels": fulldataset.gestures, "imagesize": imagesize}, "cnn.pth")

#fine tuning
finetuneepochs = 5
finetunelearningrate = 0.0001 #smaller learning rate bc fine tuning should make smaller updates
personaltestsize = int(len(personaldataset) * 0.15)
personalvalidationsize = int(len(personaldataset) * 0.15)
personaltrainsize = len(personaldataset) - personalvalidationsize - personaltestsize
personaltraindataset, personalvalidationdataset, personaltestdataset = random_split(personaldataset, [personaltrainsize, personalvalidationsize, personaltestsize], generator = torch.Generator().manual_seed(42)) #splits dataset with reproducibility
#load fine tuning dataloaders
personaltrainloader = DataLoader(personaltraindataset, batch_size = batch, shuffle = True)
personalvalidationloader = DataLoader(personalvalidationdataset, batch_size = batch, shuffle = False)
personaltestloader = DataLoader(personaltestdataset, batch_size = batch, shuffle = False)

print("fine tuned split:")
print("personal train samples: ", personaltrainsize)
print("personal validation samples: ", personalvalidationsize)
print("personal test samples: ", personaltestsize)

finetuneoptimizer = torch.optim.Adam(model.parameters(), lr=finetunelearningrate) #new adam optimizer with smaller lr

finetunetrainlosshistory = []
finetunevalidationlosshistory = []
finetunetrainaccuracyhistory = []
finetunevalidationaccuracyhistory = []
finetunevalidationf1history = []

for epoch in range(finetuneepochs):
    model.train()
    runningloss = 0.0
    correctpredictions = 0
    totalsamples = 0
    
    for images, labels in personaltrainloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        finetuneoptimizer.zero_grad()
        loss.backward()
        finetuneoptimizer.step()
        runningloss += loss.item()
        predictions = torch.argmax(outputs, dim = 1)
        correctpredictions += (predictions == labels).sum().item()
        totalsamples += labels.size(0)
    trainloss = runningloss / len(personaltrainloader)
    trainingaccuracy = correctpredictions / totalsamples

    validationloss, validationaccuracy, validationf1, yvalidationtrue, yvalidationprediction = evaluate(model, personalvalidationloader)
    finetunetrainlosshistory.append(trainloss)
    finetunevalidationlosshistory.append(validationloss)
    finetunetrainaccuracyhistory.append(trainingaccuracy)
    finetunevalidationaccuracyhistory.append(validationaccuracy)
    finetunevalidationf1history.append(validationf1)

    print(f"fine tuned epoch {epoch+1:02d}/{finetuneepochs}. "f"train loss: {trainloss:.4f}. "f"validation loss: {validationloss:.4f}. "f"training accuracy: {trainingaccuracy:.4f}. "f"validation accuracy: {validationaccuracy:.4f}. "f"validation macrof1: {validationf1:.4f}")

#fine tuned evaluation
finetunetestloss, finetunetestaccuracy, finetunedtestf1, yfinetunetrue, yfinetunepredictions = evaluate(model, personaltestloader)
finetunematrix = confusionmatrix(yfinetunetrue, yfinetunepredictions, numclasses)

print("final fine tuned personal test metrics:")
print(f"fine tuned personal test loss: {finetunetestloss:.4f}")
print(f"fine tuned personal test accuracy: {finetunetestaccuracy:.4f}")
print(f"fine tuned personal test macrof1: {finetunedtestf1:.4f}")
print("fine tuned person test confusion matrix:")
print(finetunematrix)

#fine tuned plots
finetuneepoch = range(1, finetuneepochs + 1)
#training vs validation loss
plt.figure()
plt.plot(finetuneepoch, finetunetrainlosshistory, label = "fine tuned train loss")
plt.plot(finetuneepoch, finetunevalidationlosshistory, label = "fine tuned validation loss")
plt.title("cnn fine tuned: training vs validation Loss")
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("cnnfinetunedlosscurve.png")
plt.show()
#training vs validation accuracy
plt.figure()
plt.plot(finetuneepoch, finetunetrainaccuracyhistory, label = "fine tuned train accuracy")
plt.plot(finetuneepoch, finetunevalidationaccuracyhistory, label = "fine tuned validation accuracy")
plt.title("cnn fine tuned: training vs validation accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("cnnfinetunedaccuracycurve.png")
plt.show()
#confusion matrix
plt.figure(figsize = (8, 6))
plt.imshow(finetunematrix, cmap = "Blues")
plt.title("cnn fine tuned: personal test confusion matrix")
plt.xlabel("predicted class")
plt.ylabel("true class")
plt.xticks(np.arange(numclasses), fulldataset.gestures, rotation = 45)
plt.yticks(np.arange(numclasses), fulldataset.gestures)

for i in range(numclasses):
    for j in range(numclasses):
        plt.text(j, i, finetunematrix[i, j], ha = "center", va = "center")
plt.colorbar()
plt.tight_layout()
plt.savefig("cnnfinetunedpersonalconfusionmatrix.png")
plt.show()
torch.save({"learnedweights": model.state_dict(), "labels": fulldataset.gestures, "imagesize": imagesize}, "cnn_finetuned.pth") #saves fine tuned cnn model
print("saved fine tuned model to cnn_finetuned.pth")

#plots
epochrange = range(1, epochs + 1)
#training vs validation loss
plt.figure()
plt.plot(epochrange, trainlosshistory, label = "train loss")
plt.plot(epochrange, validationlosshistory, label = "validation loss")
plt.title("cnn: training vs validation Loss")
plt.xlabel("epoch")
plt.ylabel("cross entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("cnnlosscurve.png")
plt.show()
#training vs validation accuracy
plt.figure()
plt.plot(epochrange, trainaccuracyhistory, label = "train accuracy")
plt.plot(epochrange, validationaccuracyhistory, label = "validation accuracy")
plt.title("cnn: training vs validation accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("cnnaccuracycurve.png")
plt.show()
#validation macrof1
plt.figure()
plt.plot(epochrange, validationf1history, label = "validation macrof1")
plt.title("cnn: validation macrof1")
plt.xlabel("epoch")
plt.ylabel("macro Ff1")
plt.legend()
plt.tight_layout()
plt.savefig("cnnmacrof1curve.png")
plt.show()
#final evaluation
plt.figure(figsize = (8, 6))
plt.imshow(testconfusionmatrix, cmap = "Blues")
plt.title("cnn: kaggle test confusion matrix")
plt.xlabel("predicted class")
plt.ylabel("true class")
plt.xticks(np.arange(numclasses), fulldataset.gestures, rotation = 45)
plt.yticks(np.arange(numclasses), fulldataset.gestures)

for i in range(numclasses):
    for j in range(numclasses):
        plt.text(j, i, testconfusionmatrix[i, j], ha = "center", va = "center")
plt.colorbar()
plt.tight_layout()
plt.savefig("cnntestconfusionmatrix.png")
plt.show()
plt.figure(figsize = (8, 6))
plt.imshow(personalconfusionmatrix, cmap = "Blues")
plt.title("cnn: personal dataset confusion matrix")
plt.xlabel("predicted class")
plt.ylabel("true class")
plt.xticks(np.arange(numclasses), fulldataset.gestures, rotation = 45)
plt.yticks(np.arange(numclasses), fulldataset.gestures)

for i in range(numclasses):
    for j in range(numclasses):
        plt.text(j, i, personalconfusionmatrix[i, j], ha = "center", va = "center")

plt.colorbar()
plt.tight_layout()
plt.savefig("cnnpersonalconfusionmatrix.png")
plt.show()
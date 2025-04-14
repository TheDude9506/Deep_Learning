import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torchvision import models,transforms,datasets
import scipy.io
import numpy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import math
import torch.nn.functional as F
import random


# #Set device to reduce training time
# DEVICE=(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# torch.set_default_device(DEVICE)
# print(f"Using {DEVICE} device")

#Convert Array numpy into tensor
class ArrayToTensor(object):
    """
    Trasform array NumPy into PyTorch tensor.
    """
    def __call__(self, array):
        #convert array into image
        img = Image.fromarray(array)
        # Convert image into Pytorch tensor
        tensor = transforms.ToTensor()(img)
        return tensor

class LympDataset(Dataset):
    """Lymphoma Dataset

    Args:
        labels(list): list of labels
        patterns(list): list of patterns
        transform(callable): transform to apply to the patterns
    """
    def __init__(self, labels, data, transform=None):
        self.labels= torch.tensor(labels, dtype=torch.long)
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): #This method is used to obtain a sample from the dataset, given an index
        return self.transform(self.data[idx]), self.labels[idx] - 1

#Define the transformation of the data augmentation
transform=transforms.Compose([
    ArrayToTensor(),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # Randomly crop and resize the image
    transforms.RandomHorizontalFlip(p=0.5), # Perform a horizontal flip with a 50% chance
    transforms.RandomVerticalFlip(p=0.5),   # Perform a vertical flip with a 50% chance
    transforms.RandomRotation(degrees=30),  # Randomly rotates the image up to +/- 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly changes brightness, constrast,saturation and hue
    transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2), # Randomly applies perspective transformations
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),  # Randomly applies affine transformations
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erases a rectangular region of the images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizes the image to the ImageNet mean and standard deviation
])

# load .mat file
dataset = scipy.io.loadmat('DatasColor_29.mat')

#train dataset
train_dataset = LympDataset(dataset['DATA'][0][1][0],dataset['DATA'][0][0][0], transform)

#test dataset
test_dataset = LympDataset(dataset['DATA'][0][1][0], dataset['DATA'][0][0][0], transform)
#Create a Dataloader instance --> Dataloader is a class uses for load and manage dataset during training
#                             --> Dataloader subdivide my dataset into batch
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, generator = torch.Generator(device = 'cuda'))
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True, generator = torch.Generator(device = 'cuda'))

#Load pretrained ResNet model
model = models.resnet50(pretrained=True)

#I replace the last layer with a new layer adapted to the number of classes in my dataset
num_classi = 3 #number of classes in my dataset
num_features=model.fc.in_features
"""
   I adapt the ResNet model for a 'num_classi' class classification problem, 
   I set the new classifier to ensure that the number of input features corresponds 
   to the number of outputs produced by the previous layers of the network
"""
model.fc=nn.Linear(num_features, num_classi)

#Loss and Optimization function
loss_fn=nn.CrossEntropyLoss()
opt_fn=optim.Adam(model.parameters(), lr=0.001)

# Initialize tensorboard logger
writer = SummaryWriter()

torch.utils.data.DataLoader(
    dataset = 'DatasColor_29.mat',
    generator = torch.Generator(device = 'cuda')
)


DEVICE = 'cuda'
torch.set_default_device('cuda')
print(DEVICE)

#TRAINING THE MODEL
# epochs = 30   # <-- IF  YOU HAVE "CUDA" DEVICE (EX: NVIDIA GPU) IMPROVE THIS VARIABLE
# iterations=len(train_dataloader)
# for epoc in range(epochs):                       #for each epoch
#     for i, data in enumerate(train_dataloader):  #for each batch
#
#         inputs,labels= data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda'))      #loop that iterates over each batch in the DataLoader
#                                                                    #inputs=This variable contains a batch of training data, which are tensors representing the model inputs.
#                                                                    #labels=This variable contains labels corresponding to the inputs, if any.
#                                                                    #The labels are tensors that represent the reference classes for each data in the batch.
#         opt_fn.zero_grad()
#         model = model.to(torch.device('cuda'))#is used to reset the gradients of the model parameters before performing a new backward pass
#         outputs = model(inputs.to(torch.device('cuda')))          #passes input data through the model to obtain model predictions.
#         loss = loss_fn(outputs, labels)  #This statement calculates the loss (or error) between the model predictions (outputs) and the reference labels (labels).
#         loss.backward()                  #performs the backward pass to compute the gradients of all variables that have requires_grad=True.
#                                          # This calculates the gradients of the loss function with respect to the model parameters.
#         opt_fn.step()                    #updates the model weights using the Adam algorithm
#         #writer.add_scalar('Loss/train', loss.item(), epoc * iterations + i + 1)  #logger --> give me grafic
#         print(f'epoch[{epoc+1}/{epochs}], loss = {loss.item():.4f}')
# print()

FILE = 'model30.pth'
# torch.save(model, FILE)

loaded_model = torch.load(FILE)
#loaded_model.eval()
#TESTING

transform1 = transforms.Compose([
    ArrayToTensor() ])
train_dataset_without = LympDataset(dataset['DATA'][0][1][0], dataset['DATA'][0][0][0], transform1)

print(train_dataset)
print(train_dataset_without)
train_dataloader_without = DataLoader(train_dataset_without, batch_size = 64, shuffle = True, generator = torch.Generator(device = 'cuda'))
print(train_dataloader)
print(train_dataloader_without)
class_labels = []
class_preds = []
test_loss = 0.0
iterations = len(train_dataloader_without)
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    n_samples = 0 # numbers of outputs of nn
    n_correct = 0 # numbers of correct outputs of nn (when output == label)
    for i, data in enumerate(train_dataloader_without):
        inputs, labels = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda'))
        loaded_model1 = loaded_model.to(torch.device('cuda')) # load saved model
        outputs = loaded_model1(inputs)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0) # count nuber of outputs
        n_correct += (predicted == labels).sum().item() # count number of correct outputs

        preds = [F.softmax(output, dim = 0 ) for output in outputs] # creating the list of all outputs of nn

        class_preds.append(preds)
        class_labels.append(predicted)
        test_loss += criterion(outputs, labels).item() * labels.size(0)

    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)
    acc = 100.0 * n_correct / n_samples
    test_loss /= len(train_dataloader_without.dataset)
    print(f'Accuracy of the network on the {n_samples} train images wirhout augmentation: {acc} %')
    print(f'Test loss: {test_loss}')

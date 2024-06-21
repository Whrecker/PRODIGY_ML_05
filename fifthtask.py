import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from torchvision import models
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import OrderedDict
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_predictions(my_model, data):
    pred_probs = []
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0)
            sample = sample.to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob)
    return torch.stack(pred_probs)

classes = open("/kaggle/input/please/food-101 - Copy/food-101/meta/classes.txt", 'r').read().splitlines()
classes_21 = classes
calorie_classes = pd.read_excel("/kaggle/input/aahhhh/pain.xlsx", sheet_name="pain")
classes = []
calories = {}
temp = []

for i in classes_21:
    for j in range(len(calorie_classes['name'])):
        if i in calorie_classes['name'][j] and i not in classes:
            calories[calorie_classes['name'][j]] = calorie_classes['Calories'][j]
            temp.append(calorie_classes['name'][j])
            classes.append(i)

def prep_df():
    global calories
    img_path = "/kaggle/input/please/food-101 - Copy/food-101/images/"
    full_path = []
    for i in classes:
        full_path.append(img_path + i)
    imgs = []
    count = -1
    labelss = []
    for i in range(len(full_path)):
        count = count + 1
        calories[count] = calories[temp[i]]
        del calories[temp[i]]
        for j in os.listdir(full_path[i]):
            imgs.append(full_path[i] + "/" + j)
            labelss.append(count)
    aimgs = np.array(imgs)
    imgs = pd.DataFrame(imgs)
    imgs['path'] = aimgs
    imgs['label'] = labelss
    imgs = shuffle(imgs)
    return imgs

imgs = prep_df()
train_imgs = imgs[:int(len(imgs) * .7)]
test_imgs = imgs[int(len(imgs) * .7):]

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Food(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.dataframe['path'].iloc[idx]
        image = Image.open(img_name).convert('RGB')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = self.dataframe['label'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = Food(train_imgs, transform=train_transforms)
test_dataset = Food(test_imgs, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

weights = models.DenseNet201_Weights.IMAGENET1K_V1
model = models.densenet201(weights=weights)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(1920, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 101),
)

model.classifier = classifier
model.to(device)

num_epochs = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
model = model.to(device)

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        images, labels = X.to(device), y.to(device)
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            images, labels = X.to(device), y.to(device)
            test_pred_logits = model(images)
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += ((test_pred_labels == labels).sum().item() / len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            f"\n\n=============================\n"
        )
        if test_acc > 0.95:
            break
    return model

model= train(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device)

def evaluate(model, dataloader):
    random_idx = np.random.randint(0, len(dataloader))
    with torch.inference_mode():
        model.eval()
        n_correct = 0
        n_samples = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, 1), 1)
            preds = np.array([pred.cpu() if pred < 20 else 20 for pred in preds])
            labels = np.array([label.cpu() if label < 20 else 20 for label in labels])
            n_samples += labels.shape[0]
            n_correct += (preds == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(acc)

evaluate(model, test_loader)

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_dataset), k=10):
    test_samples.append(sample)
    test_labels.append(label.item())

model.eval()
pred_probs = make_predictions(model, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

for i in test_labels:
    if i in pred_classes:
        print("it is label", i, "and the calories are", calories[i])

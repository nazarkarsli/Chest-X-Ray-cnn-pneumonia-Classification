import sns
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from PIL import Image
import os
import seaborn as sns
from tqdm import tqdm
from torchvision.datasets import ImageFolder
batch_size = 64
image_size = 28*28
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )) #xray görüntüleri tek bir kanal içerir çnk siyah-beyazlar, rgb renk bilgisi yok.
])
data_dir = "Kendi_Kodlarim/data/ChestXRay/chest_xray"
train_dataset = ImageFolder(root="./data", transform=transform)
val_dataset   = ImageFolder(root="./data", transform=transform)
test_dataset  = ImageFolder(root="./data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3, padding=1) #giriş 1 kanal
        self.pool =nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*56*56,128)
        self.fc2 = nn.Linear(128,2) #2 sınıf : normal, pnömoni
        self.dropout=nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 → 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 → 56
        x = x.view(-1, 32 * 56 * 56)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
model =CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs= 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images,labels in tqdm(train_loader, desc =f"Epoch {epoch+1} "):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images,labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _ , predicted = torch.max(outputs, 1)
        correct+= (predicted==labels).sum().item()
        total+= labels.size(0)
print(f"Validation doğruluk: {100 * correct / total: .2f}%")
all_preds=[]
all_labels=[]
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_loader.dataset.classes, yticklabels=train_loader.dataset.classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')
plt.show()


print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

torch.save(model.state_dict(), "chest_xray_cnn.pth")
print("Model başarıyla kaydedildi: chest_xray_cnn.pth")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, gzip
from urllib.request import urlretrieve

#loading the Dataset URLs 
TRAIN_IMAGES_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz'
TRAIN_LABELS_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz'
TEST_IMAGES_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz'
TEST_LABELS_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz'

# defining the Download Function 
def download_data(directory="./data"):
    os.makedirs(directory, exist_ok=True)
    for url in [TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL]:
        filename = os.path.join(directory, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(url, filename)

# File Loader 
def load_data(directory="./data", train=True):
    if train:
        img_path = os.path.join(directory, 'train-images-idx3-ubyte.gz')
        lbl_path = os.path.join(directory, 'train-labels-idx1-ubyte.gz')
    else:
        img_path = os.path.join(directory, 't10k-images-idx3-ubyte.gz')
        lbl_path = os.path.join(directory, 't10k-labels-idx1-ubyte.gz')

    with gzip.open(img_path, 'rb') as img_f:
        images = np.frombuffer(img_f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(lbl_path, 'rb') as lbl_f:
        labels = np.frombuffer(lbl_f.read(), np.uint8, offset=8)

    return images.astype(np.float32) / 255.0, labels

# Custom Dataset Class 
class MyFashionDataset(Dataset):
    def __init__(self, root="./data", train=True):
        download_data(root)
        self.images, self.labels = load_data(root, train)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).unsqueeze(0)
        label = self.labels[idx]
        return image, label

# intializing the CNN Model 
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# Training the model and Evaluation  
def train_model(model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Model Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Accuracy = {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch
            }, "best_model.pt")
            print(f" Saved best model with accuracy: {accuracy:.2f}%")

def evaluate_model(model_path="best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_dataset = MyFashionDataset(train=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"\n Loaded model from {model_path}")
        print(f"Final Evaluation Accuracy: {accuracy:.2f}%")

        # Show sample predictions
        print("\nSample Predictions:")
        for i in range(5):
            image, label = test_dataset[i]
            img_input = image.unsqueeze(0).to(device)
            pred = model(img_input).argmax(1).item()
            print(f"Sample {i+1}: Predicted = {pred}, Actual = {label}")

    except FileNotFoundError:
        print(f" Model file '{model_path}' not found.")

# Starting the Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MyFashionDataset(train=True)
    test_dataset = MyFashionDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNNClassifier().to(device)
    print("\n Starting Training...\n")
    train_model(model, train_loader, test_loader, device)

    print("\n Evaluating Trained Model...\n")
    evaluate_model("best_model.pt")

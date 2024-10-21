import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

# Function to load an image and its label
def load_image_and_label(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Get the corresponding label file path
    label_path = image_path.rsplit('.', 1)[0] + '.txt'
    
    # Read the label from the text file
    try:
        with open(label_path, 'r') as file:
            label = int(file.read().strip())
    except:
        print(f"Warning: Couldn't read label for {image_path}. Assuming 0.")
        label = 0
    
    return image, label

# Function to prepare the dataset
def prep_dataset(folder_path, transform):
    dataset = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image, label = load_image_and_label(image_path)
            if transform:
                image = transform(image)
            dataset.append((image, label))
    return dataset

# Simple CNN model
class SimpleArmorClassifier(nn.Module):
    def __init__(self):
        super(SimpleArmorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.sigmoid(self.fc(x))
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs} completed')

# Main execution
if __name__ == '__main__':
    # Set up data transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Prepare datasets
    train_data = prep_dataset('C:/Users/Akshaj/Documents/train', data_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize the model
    model = SimpleArmorClassifier()
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer)
    
    # Save the trained model
    torch.save(model.state_dict(), 'armor_classifier.pth')
    
    print("Training completed and model saved!")
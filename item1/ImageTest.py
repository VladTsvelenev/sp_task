import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
import warnings 

warnings.filterwarnings('ignore')

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #150*150*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3,3)) 
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(3,3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*17*17, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)

    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

dictionary = {0: 'Jeans',
              1: 'Sofa',
              2: 'T-shirt',
              3: 'Television',
              4: 'Other'}

image = Image.open('test.jpg')

transform1 = transforms.Compose([         
            v2.ToTensor(),
            v2.Resize([150, 150])])

image_transformed = transform1(image)
plt.imshow(image_transformed.permute(1,2,0).data.cpu().numpy())
plt.show()

model = ConvNet()
model.load_state_dict(torch.load('test.pt'))
model.eval()
with torch.no_grad():
    model_output = model(image_transformed.reshape((1, 3, 150, 150)))

print(dictionary[np.argmax(model_output.data.cpu().numpy())])
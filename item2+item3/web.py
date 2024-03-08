from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import warnings 
import io
import base64

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

    
def classify_image(image):
    global model
    transform_tens = transforms.Compose([         
            v2.ToTensor(),
            v2.Resize([150, 150])])
    transform_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_transformed = transform_tens(image)
    with torch.no_grad():
        model_output = model(image_transformed.reshape((1, 3, 150, 150)))
    norm_img = transform_norm(image_transformed).unsqueeze(0)
    integrated_gradients = IntegratedGradients(model)
    im_at = integrated_gradients.attribute(norm_img, target=3, n_steps=200)
    map = LinearSegmentedColormap.from_list('custom blue', [(0, '#000000'), (0.3, '#0000ff'), (1, '#0000ff')], N=256)
    figure, _ = viz.visualize_image_attr(np.transpose(im_at.squeeze().cpu().detach().numpy(), (1,2,0)),
                                np.transpose(image_transformed.squeeze().cpu().detach().numpy(), (1,2,0)),
                                method='heat_map', cmap=map, use_pyplot=False)
    result_image = io.BytesIO()
    figure.savefig(result_image, format='png', bbox_inches='tight', pad_inches=0)
    result_image.seek(0)
    result_text = dictionary[np.argmax(model_output.data.cpu().numpy())]
    return result_image, result_text


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)

dictionary = {0: 'Jeans', 1: 'Sofa', 2: 'T-shirt', 3: 'Television', 4: 'Other'}
model = ConvNet()
model.load_state_dict(torch.load('test.pt'))
model.eval()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            image = Image.open(file)
            result_image, result_text = classify_image(image)
            result_image_base64 = base64.b64encode(result_image.read()).decode()
            return jsonify({'result_text': result_text, 'result_image_base64': result_image_base64})
        else:
            return jsonify({'error': 'Invalid image file'})

    return jsonify({'error': 'No image provided'})

if __name__ == '__main__':
    app.run(debug=True)
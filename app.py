# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:02:53 2024

@author: TEJA
"""

from flask import Flask, render_template, request
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
#from utils_inceptionresnetv2 import InceptionResNetV2
from models import FaceNetModel
from PIL import Image
import base64
import io
mtcnn = MTCNN()

app = Flask(__name__)

preprocess=transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomResizedCrop(199),
    #transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

# Function to load your model
def load_model():
    # Load your PyTorch model here
    # Example:
    
    # model = YourModel()
    # model.load_state_dict(torch.load('path_to_your_model.pth', map_location=torch.device('cpu')))
    # model.eval()
    # return model
    
    model=FaceNetModel()
    

    best_model='E:/programmer me/unlearning/pins_upto_epoch_65/kaggle/working/log/best_state.pth'
    unlearned_model='E:/programmer me/unlearning/pins_upto_epoch_65/kaggle/working/log/last_checkpoint.pth'
    checkpoint=torch.load(unlearned_model,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# Function to make predictions
def predict(image):
    # Load your model
    model = load_model()
    #aligned=mtcnn(image)
    # Perform inference
    image=preprocess(image)
    image=torch.unsqueeze(image,dim=0)
    with torch.no_grad():
        embeddings = model.forward_classifier(image)
        #embeddings[0][19]=0.5
    probabilities = torch.softmax(embeddings, dim=1)
    print(probabilities)

    return embeddings
    #return probabilities

@app.route('/predict',methods=['POST'])
def find_image():
    prediction = None
    img_str = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=prediction)

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=prediction)

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Make predictions
        probabilities = predict(image)

        # Convert tensor to numpy array
        probabilities = probabilities.detach().numpy()

        # Display prediction (example: using argmax)
        pred_1=str(probabilities.argmax())
        prediction = str(probabilities)
        prediction+=' '+pred_1
        # Convert image to base64 string for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', prediction=prediction, img=img_str)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_str = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=prediction)

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=prediction)

        image = Image.open(io.BytesIO(file.read()))

        # Make predictions
        probabilities = predict(image)

        # Convert tensor to numpy array
        probabilities = probabilities.detach().numpy()

        # Display prediction (example: using argmax)
        prediction = str(probabilities.argmax())

        # Convert image to base64 string for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', prediction=prediction, img=img_str)

if __name__ == '__main__':
    app.run(debug=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self, num_kernels, kernel_size):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, num_kernels, kernel_size=(kernel_size,kernel_size), padding=1), # (28-kernel_size+3)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (13, 13)
            
            nn.Conv2d(num_kernels, num_kernels, kernel_size=(kernel_size-1,kernel_size-1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (6, 6)
            
            nn.Conv2d(num_kernels, num_kernels, kernel_size=(kernel_size-2,kernel_size-2), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (3, 3)
        )
        
        self.dropout =  nn.Dropout(.5)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * num_kernels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10) 
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dropout(x)
        output = self.fc_layers(x)
        return output

def get_probs(output):
    return F.softmax(output[0], dim=0)

def predict(img_array):
    model = CNN(num_kernels=16, kernel_size=5)
    model_path = os.path.join(os.path.dirname(__file__), "pytorch_model/cnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.view(-1, 1, 28, 28) 

    with torch.no_grad():
        output = model(img_tensor)
        return get_probs(output)
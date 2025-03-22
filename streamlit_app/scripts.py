import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(16*7*7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

def get_probs(output):
    return F.softmax(output[0], dim=0)

def predict(img_array):
    model = CNN()
    model_path = os.path.join(os.path.dirname(__file__), "pytorch_model/cnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.view(-1, 1, 28, 28) 

    with torch.no_grad():
        output = model(img_tensor)
        return get_probs(output)
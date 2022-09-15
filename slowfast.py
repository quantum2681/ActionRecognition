import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import params

model_path = params['model_path']
resnet50 = torch.load(model_path)
resnet50.load_state_dict(resnet50.state_dict())

if __name__ == "__main__":
    input_tensor = [torch.rand(1, 3, 8, 224, 224), torch.rand(1, 3, 32, 224, 224)]
    model = resnet50
    output = model(input_tensor)
    # print(output.size())

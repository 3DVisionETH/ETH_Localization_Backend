import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json 
from pathlib import Path

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    print("WARNING - no GPU found, training will be slow") 

class Automata(nn.Module):
    def __init__(self, program_in, program_out, memory_in, memory_out, k=1):
        super().__init__()

        self.program_in = program_in
        self.program_out = program_out 
        self.memory_in = memory_in
        self.memory_out = memory_out 

        self.modify_program = nn.Sequential(
            nn.Conv2d(program_in, program_out, (3,3), padding=(1,1)),
            nn.LeakyReLU(),
        )

        self.mask = nn.Sequential(
            nn.BatchNorm2d(program_out),
            nn.Conv2d(program_out, program_out, (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(program_out, 6*memory_in, (1,1))
        )

        self.update = nn.Sequential(
            nn.BatchNorm2d(memory_in),
            nn.Conv2d(memory_in,memory_out, (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(memory_out,memory_out, (1,1)),
            nn.BatchNorm2d(memory_out),
        )
        self.activation = torch.nn.LeakyReLU()
        self.quantize_program = torch.nn.Softmax(dim=1) # torch.nn.Identity() #torch.nn.Tanh() 
        self.quantize_memory = torch.nn.Softmax(dim=1) #torch.nn.Identity() #torch.nn.Tanh() 

        self.k = k

    def forward(self, program, memory):
        b,c,h,w = memory.shape
        res_program = self.program_in == self.program_out 
        res_memory = self.memory_in == self.memory_out

        memory_in, memory_out = self.memory_in, self.memory_out
        program = program + self.modify_program(program) if res_program else self.modify_program(program) 

        mask = self.mask(self.quantize_program(program)).reshape((b,6,memory_in,h,w))
        xn = mask[:,0] 
        xm = mask[:,1] 
        yn = mask[:,2] 
        ym = mask[:,3] 
        c = mask[:,4]
        bias = mask[:,5]

        for i in range(self.k):
            res = memory 

            accum_y = yn* torch.cat([memory[:,:,0:1,:], memory[:,:,:-1,:]], dim=2) + ym * torch.cat([memory[:,:,1:,:], memory[:,:,-1:,:]],dim=2)
            accum_x = xn* torch.cat([memory[:,:,:,0:1], memory[:,:,:,:-1]], dim=3) + xm * torch.cat([memory[:,:,:,1:], memory[:,:,:,-1:]],dim=3)
            accum_c = c*memory
            accum = accum_x + accum_y + accum_c + bias

            memory = res + self.update(accum) if res_memory else self.update(accum)
            memory = self.activation(memory)
            memory = self.quantize_memory(memory)

        return program, memory

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.max_pool(x)
        return x

class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(2,2), stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_image0 = ConvBlockDown(1,4)
        self.down_image1 = ConvBlockDown(4,8)
        self.down_image2 = ConvBlockDown(8,16)
        self.down_image3 = ConvBlockDown(16,32)

        self.up_image3 = ConvBlockUp(32,16)
        self.up_image2 = ConvBlockUp(32,24)
        self.up_image1 = ConvBlockUp(32,28)
        self.up_image0 = ConvBlockUp(32,16)

        self.final = nn.Sequential(
            nn.Conv2d(17,16,(3,3),padding=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.down_image0(x)
        x1 = self.down_image1(x0)
        x2 = self.down_image2(x1)
        x3 = self.down_image3(x2)

        x4 = torch.cat([x2, self.up_image3(x3)], dim=1)
        x5 = torch.cat([x1, self.up_image2(x4)], dim=1)
        x6 = torch.cat([x0, self.up_image1(x5)], dim=1)
        x7 = torch.cat([x, self.up_image0(x6)], dim=1)
        x8 = self.final(x7)

        return x8


class AutomataBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, memory_in, memory_out):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.automata = Automata(in_channels, out_channels, memory_in, memory_out)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x, m):
        x, m = self.automata(x, m)
        x, m = self.max_pool(x), self.max_pool(m)
        return x, m

class AutomataBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, memory_in, memory_out):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.automata = Automata(in_channels, out_channels, memory_in, memory_out, k=2)
        self.up_conv_program = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv_memory = nn.ConvTranspose2d(memory_out, memory_out, kernel_size=2, stride=2)

    def forward(self, x, m):
        x, m = self.automata(x, m)
        return self.up_conv_program(x), self.up_conv_memory(m)

class DiffFlood(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.ModuleList([AutomataBlockDown(16,16, 1 if i==0 else 2,2) for i in range(4)])
        self.up = nn.ModuleList([AutomataBlockUp(16 if i==0 else 32,16,2,2) for i in range(4)])
        
        self.refine0 = Automata(32,16,2,2,k=1)
        self.refine1 = Automata(16,16,2,2,k=8)

    def forward(self, x):
        b,c,h,w = x.shape
        m = torch.ones((b,1,h,w), device=x.device)

        inputs = [x]
        for i in range(len(self.down)):
            x, m = self.down[i](x, m)
            inputs.append(x)    
        inputs = inputs[::-1]

        for i in range(len(self.up)):
            x, m = self.up[i](x, m)
            x = torch.cat([x, inputs[i+1]],dim=1)

        x,m = self.refine0(x,m)
        x,m = self.refine1(x,m)
        return x,m

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.diff_flood = DiffFlood()
        self.final = nn.Sequential(
            nn.Conv2d(2,2,(1,1)),
            nn.ReLU(),
            nn.Conv2d(2,1,(1,1)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x,m = self.diff_flood(x)
        return self.final(m)

def load_room_segmentation_model():
    model_file = Path(__file__).parent / "room_model.pt"
    print("Loading ", model_file)

    model = Model()
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model

def select_device():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    elif use_mps:
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

def generate_mask(model, image, max_size=1024):
    assert len(image.shape) == 2, "Expecting gray-scale image"

    scale = max(image.shape[:2]) / max_size
    height, width = image.shape[:2]

    round = 256
    image = cv2.resize(image, (max_size,max_size)) #(int(width*scale) // round * round, int(height*scale) // round * round))

    image_torch = 1.0 - torch.tensor(image.astype(np.float32) / 255).to(device)
    model = model.to(device)
    
    mask = model(image_torch[None,None])[0,0]
    mask = mask.cpu().detach().numpy()

    plt.imshow(mask)
    plt.show()
    
    mask = cv2.resize(mask, (width,height))
    return mask

device = select_device()

if __name__ == "__main__":
    # Quick test for debugging
    # Use scripts/preprocess.py for the command line interface

    path = "data/segmentation/gt/CAB_Floor_E.png"
    output_dir = Path("data/segmentation/output/CAB_E")

    model = load_room_segmentation_model()

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = generate_mask(model, image)

    cv2.imwrite(output_dir / "room_mask.png", mask)
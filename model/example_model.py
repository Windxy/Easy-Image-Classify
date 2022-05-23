import torch
from torch import nn

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel,self).__init__()
        self.conv = nn.Sequential(
            # C × H × W
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=5,stride=1,padding=2), # 11 * 11 Conv(96),stride 1
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3,2),  # kernel_size=3,stride=2

            nn.Conv2d(96,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(256,384,3,1,1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384,384,3,1,1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3,10), # full connection layer
        )
    def forward(self,img):
        feature = self.conv(img)
        return self.fc(feature)
        # return self.fc(feature.view(img.shape[0],-1))

if __name__ == '__main__':
    from torchstat import stat

    net = ExampleModel()
    stat(net,(3, 32,32))
    X = torch.rand(1, 3, 32, 32)
    print(net(X).shape)
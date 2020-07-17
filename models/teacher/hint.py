import torch
import torch.nn as nn

# Hint Nets
# settings
hint_cfg = {
    'Hint5':[256, 512, 'M', 1024, 1024, 'M', 512],
    'Hint7':[256, 512, 'M', 1024, 1024, 'M', 2048, 'M', 1024, 512, 'M'],
    'Hint9':[128, 256, 'M', 512, 1024, 'M', 2048, 1024, 'M', 512, 512, 'M'],
}

# model
class Hint(nn.Module):
    def __init__(self, hint_name, num_classes=10):
        super(Hint, self).__init__()
        self.features = self._make_layers(hint_cfg[hint_name])
#         self.classifier = nn.Linear(2048, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for h in cfg:
            if h == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else :
                layers += [nn.Conv2d(in_channels, h, kernel_size=3, padding=1),
                            nn.BatchNorm2d(h),
                            nn.ReLU(inplace=True)]
                in_channels = h
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Hints
def Hint5(num_classes=10):
    return Hint('Hint5', num_classes)

def Hint7(num_classes=10):
    return Hint('Hint7', num_classes)

def Hint9(num_classes=10):
    return Hint('Hint9', num_classes)

import torch
import torch.nn as nn

# VGGs
# settings
cfg = {
    'VGG11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# model
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, name=None):
        super(VGG, self).__init__()
        self.model_name = name
        
        self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        pool = self.features(x)
        out = pool.view(pool.size(0), -1)
        out = self.classifier(out)
        return pool, out
    
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

# VGGs
def VGG11(num_classes=10):
    return VGG('VGG11', num_classes, name='VGG11')

def VGG13(num_classes=10):
    return VGG('VGG13', num_classes, name='VGG13')

def VGG16(num_classes=10):
    return VGG('VGG16', num_classes, name='VGG16')

def VGG19(num_classes=10):
    return VGG('VGG19', num_classes, name='VGG19')
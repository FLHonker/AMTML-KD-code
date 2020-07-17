# ---------------------
#   Student Nets
# ---------------------
import torch.nn as nn
import torch.nn.functional as F


# Linear Net-1
class fcNet(nn.Module):
    def __init__(self):
        super(StudentNet1, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc3(x)
        return x
    
import torch.nn as nn
import torch
import torch.nn.functional as F
from .adaptive_conv import Depthwise


class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size, pad=False):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        # self.lambda1 = torch.nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)
        self.inchannel = inchannel
        self.matching_cfg = 'o2o'
        # self.node_affinity = Affinity(dim)
        self.matching_loss = nn.MSELoss(reduction='sum')
        self.with_hyper_graph = True
        self.num_hyper_edge = 3
        self.angle_eps = 1e-3
        self.conv1 = Depthwise(64, inchannel)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        if pad:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))

        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.contiguous().view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))

        out4 = self.relu4(self.fc2(out3))
        if mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            # proj = F.normalize(node_s)
            clss = self.cls_head_src(out4)

            return clss, proj

        clss = self.cls_head_src(out4)
        return clss


if __name__ == '__main__':
    model = Discriminator(100, 7, 7, 13)
    x = torch.randn((1, 100, 13, 13))
    y = model(x, mode='cross_scene')

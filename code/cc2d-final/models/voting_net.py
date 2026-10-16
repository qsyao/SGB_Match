from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

VGG = torchvision.models.vgg19

class UNet_Prototype(nn.Module):
    def __init__(self, n_channels, n_classes, non_local=False):
        super(UNet_Prototype, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  VGG(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)
        
        self.prototypes_ema = nn.Parameter(torch.randn(n_classes, 64))
        self.prototypes_ema.requires_grad = False

        if non_local:
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local
      
        self.final = nn.Conv2d(64, 64, kernel_size=1, padding=0)

    def forward_feature(self, x, get_features=False):
        _, features = self.vgg.features(x, get_features=True)

        if self.non_local:
            features[4] = self.non_local_5(features[4])
        x = self.up1(features[4], features[3])
        if self.non_local: x = self.non_local_4(x)
        x = self.up2(x, features[2])
        if self.non_local: x = self.non_local_3(x)
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)
        x = self.final(x)
        return x
    
    def set_prtotype(self, prototype, id_landmark=None):
        if id_landmark is not None:
            self.prototypes_ema.data[id_landmark] = prototype.detach().squeeze()
            return self.prototypes_ema
        self.prototypes_ema.data = prototype.detach()
        return self.prototypes_ema
    
    def update_prototypes(self, new_prototype, global_step, alpha=0.99, id_landmark=None): # 0.99
        alpha = min(1 - 1 / (global_step + 1), alpha)
        if id_landmark is not None:
            self.prototypes_ema.data[id_landmark] = alpha * self.prototypes_ema.data[id_landmark] + (1 - alpha) * new_prototype.detach().squeeze()
            return self.prototypes_ema
        self.prototypes_ema.data.mul_(alpha).add_(1 - alpha, new_prototype.detach().data)
        return self.prototypes_ema
    
    def product(self, feature):
        # import ipdb; ipdb.set_trace()
        # b, c, h, w -> b, h, w, c
        feature = feature.transpose(1, 3).transpose(1, 2)
        similarity = torch.matmul(feature, self.prototypes_ema.T)
        # similarity = similarity / torch.norm(feature, dim=-1).unsqueeze(-1) / \
        #     torch.norm(self.prototypes_ema, dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return similarity.transpose(1, 2).transpose(1, 3)
    
    def product2(self, feature, id_landmark=None):
        if id_landmark is not None:
            similarity = torch.einsum('kd,bdwh->bkwh', self.prototypes_ema[id_landmark].unsqueeze(0), feature.float())
            return similarity
        similarity = torch.einsum('kd,bdwh->bkwh', self.prototypes_ema, feature.float())
        return similarity
    
    def cal_prototype(self, feature, heatmap):
        # b, c, h, w -> b, c, h*w
        feature = feature.view(feature.size(0), feature.size(1), -1)
        # b, k, h, w -> b, k, h*w
        heatmap = heatmap.view(heatmap.size(0), heatmap.size(1), -1)
        # weighted average feature by heatmap -> b, k, c
        new_prototype = torch.matmul(heatmap, feature.permute(0, 2, 1))
        weights = heatmap.sum(dim=-1).unsqueeze(-1)
        return new_prototype / weights
    
    def forward(self, x):
        feature = self.forward_feature(x)
        heatmap = self.product(feature)
        return heatmap

class UNet_Voting(nn.Module):
    def __init__(self, n_channels, n_classes, non_local=False):
        super(UNet_Voting, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  VGG(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)


        if non_local:
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local
      
        self.final = nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)

    def forward(self, x, get_features=False):
        _, features = self.vgg.features(x, get_features=True)

        if self.non_local:
            features[4] = self.non_local_5(features[4])
        x = self.up1(features[4], features[3])
        if self.non_local: x = self.non_local_4(x)
        x = self.up2(x, features[2])
        if self.non_local: x = self.non_local_3(x)
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)
        
        if get_features: return x
        
        x = self.final(x)
        
        heatmap = F.sigmoid(x[:,:self.n_classes,:,:])
        regression_x = x[:,self.n_classes:2*self.n_classes,:,:]
        regression_y = x[:,2*self.n_classes:,:,:]

        return heatmap, regression_y, regression_x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class OutConv_Sigmoid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_Sigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    test = UNet_Pretrained(3, 57)
    wtf = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    wtf = test(wtf)
    import ipdb; ipdb.set_trace()


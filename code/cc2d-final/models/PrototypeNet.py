import torch
import torch.nn as nn
import torch.nn.functional as F

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.prototypes_ema = nn.Parameter(torch.randn(19, 448))
        self.prototypes_ema.requires_grad = False

    def set_prtotype(self, prototype):
        self.prototypes_ema.data = prototype.detach()
        return self.prototypes_ema
    
    def update_prototypes(self, new_prototype, global_step, alpha=0.99): # 0.99
        alpha = min(1 - 1 / (global_step + 1), alpha)
        self.prototypes_ema.data.mul_(alpha).add_(1 - alpha, new_prototype.detach().data)
        return self.prototypes_ema

class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape

    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)

class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)

class UNet2D(base_model):
    def __init__(self, in_channels, out_channels=1, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet2D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # self.prototypes_ema = nn.Parameter(torch.randn(19, 448))
        # self.prototypes_ema.requires_grad = False

    def forward_feature(self, x, return_extra=False):
        shape = x.shape
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))
        if return_extra:
            return x_dec[1:-1], x_dec[-1]
        else:
            output = x_dec[1:-1]
            # output = output[:-1]
            output = [F.interpolate(feature_output, size=shape[-2:], mode='bilinear', align_corners=True) for feature_output in output]
            output = torch.cat(output, dim=1)
            return output
    
    # def set_prtotype(self, prototype):
        # self.prototypes_ema.data = prototype.detach().data
        # return self.prototypes_ema
    

    
    def product(self, feature):
        # import ipdb; ipdb.set_trace()
        # b, c, h, w -> b, h, w, c
        feature = feature.transpose(1, 3).transpose(1, 2)
        similarity = torch.matmul(feature, self.prototypes_ema.T)
        # similarity = similarity / torch.norm(feature, dim=-1).unsqueeze(-1) / \
        #     torch.norm(self.prototypes_ema, dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return similarity.transpose(1, 2).transpose(1, 3)
    
    def product2(self, feature):
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

    



if __name__ == "__main__":
    model = UNet2D(in_channels=3, out_channels=1)
    fake_input = torch.randn(1, 3, 384, 384)
    output = model(fake_input)
    output = [F.interpolate(feature_output, size=fake_input.shape[-2:], mode='bilinear', align_corners=True) for feature_output in output]
    output = torch.cat(output, dim=1)
import torch
from torch import nn
from network import Resnet


class DA(nn.Module):
    def __init__(self, num_classes, trunk='resnet-50', criterion=None, cont_proj_head=0,
                 variant='D16', args=None, device=None):
        super(DA, self).__init__()

        self.args = args
        self.device = device
        # loss criterion
        self.criterion = criterion
        self.criterion_scl = nn.CosineSimilarity(dim=1).to(device)
    
        # set backbone
        self.variant = variant
        self.trunk = trunk
        final_channel = 2048
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(fs_layer=self.args.fs_layer)
            resnet.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unknown deepv3 variant: {}'.format(self.variant)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(final_channel, num_classes)
        )

        self.cont_proj_head = cont_proj_head
        if self.cont_proj_head > 0:
            self.flat = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      nn.Flatten()
            )
            self.pred = nn.Sequential(
                nn.Linear(final_channel, self.cont_proj_head, bias=False),
                nn.BatchNorm1d(self.cont_proj_head),
                nn.ReLU(inplace=True),  # hidden layer
                nn.Linear(self.cont_proj_head, final_channel)
            )

    def forward(self, x, gts=None, x_r=None, apply_fs=False):
        # encoder
        x = self.layer0[0](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_r = self.layer0[0](x_r)
        x = self.layer0[1](x)
        if self.training & apply_fs:
            x_ss = self.layer0[1](x, x_r)  # feature stylization
            with torch.no_grad():
                x_r = self.layer0[1](x_r)
        x = self.layer0[2](x)
        x = self.layer0[3](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_r = self.layer0[2](x_r)
                x_r = self.layer0[3](x_r)
            x_ss = self.layer0[2](x_ss)
            x_ss = self.layer0[3](x_ss)
        if self.training & apply_fs:
            x_tuple = self.layer1([x, x_r, x_ss])
        else:
            x_tuple = self.layer1([x])
        x_tuple = self.layer2(x_tuple)
        x_tuple = self.layer3(x_tuple)
        x_tuple = self.layer4(x_tuple)
        x = x_tuple[0]
        if self.training & apply_fs:
            x_r = x_tuple[1]
            x_ss = x_tuple[2]

        # regressor
        main_out = self.head(x)

        if self.training:
            loss_orig = self.criterion(main_out, gts.view(gts.size(0), 1))
            return_loss = [loss_orig]

            if apply_fs:
                main_out_ss = self.head(x_ss)

                if self.args.use_scl:
                    # projected features
                    assert (self.cont_proj_head > 0)
                    proj2 = self.flat(x)
                    pred2 = self.pred(proj2)
                    proj2_ss = self.flat(x_ss)
                    pred2_ss = self.pred(proj2_ss)
                    loss_scl = -(self.criterion_scl(pred2_ss, proj2.detach()).mean() +
                                 self.criterion_scl(pred2, proj2_ss.detach()).mean()) * 0.5
                    return_loss.append(loss_scl)

                if self.args.use_tl:
                    loss_tl = self.criterion(main_out_ss, gts)
                    return_loss.append(loss_tl)

                return return_loss
        else:
            return main_out


def RDA(args, num_classes, criterion, cont_proj_head, device=None):
    """
    Resnet 50 Based Network
    """
    print("Backbone : ResNet-50")
    return DA(num_classes, trunk='resnet-50', criterion=criterion, cont_proj_head=cont_proj_head, variant='D16', args=args, device=device)

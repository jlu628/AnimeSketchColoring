import torch
import torch.nn as nn

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", down=True, use_dropout=False):
        super(GeneratorBlock, self).__init__()
        down_conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
        up_conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.conv = nn.Sequential(
            down_conv if down else up_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = GeneratorBlock(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = GeneratorBlock(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = GeneratorBlock(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = GeneratorBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = GeneratorBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = GeneratorBlock(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())

        self.up1 = GeneratorBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = GeneratorBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = GeneratorBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = GeneratorBlock(features * 16, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = GeneratorBlock(features * 16, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = GeneratorBlock(features * 8, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = GeneratorBlock(features * 4, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

        
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        conv_layer = nn.Conv2d(
                in_channels*2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            )
        self.initial_layer = nn.Sequential(conv_layer, nn.LeakyReLU(0.2))

        layers = []
        layers = [DiscriminatorBlock(features[i-1], features[i], stride=2) for i in range(1, len(features) - 1)]
        layers.append(DiscriminatorBlock(features[-2], features[-1], stride=1))
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(self.initial_layer(x))

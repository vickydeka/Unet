import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, enc_cs=[1, 64, 128, 256, 512, 1024]):
        super().__init__()
        self.enc_blocklist = nn.ModuleList([Block(in_c, out_c)
                                for in_c, out_c in zip(enc_cs, enc_cs[1:])])
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feature_map = []
        for block in self.enc_blocklist:
            x = block(x)
            feature_map.append(x)
            x = self.max_pool(x)
        return feature_map

class Decoder(nn.Module):
    def __init__(self, dec_cs=[1024, 512, 256, 128, 64]):
        super().__init__()
        self.dec_cs = dec_cs
        self.transconv = nn.ModuleList([nn.ConvTranspose2d(in_c, out_c, 2, 2)
                            for in_c, out_c in zip(dec_cs[:], dec_cs[1:])])
        self.dec_block = nn.ModuleList([Block(in_c, out_c)
                            for in_c, out_c in zip(dec_cs[:], dec_cs[1:])])

    def forward(self, x, enc_feature_map):
        for i in range(len(self.dec_cs)-1):
            x = self.transconv[i](x)
            concat_ftrs = self.crop(enc_feature_map[i], x)
            x = torch.cat([x, concat_ftrs], dim=1)
            x = self.dec_block[i](x)
        return x

    def crop(self, enc_feature, x):
        delta = enc_feature.size()[2] - x.size()[2]
        delta = delta // 2
        enc_feature = enc_feature[:, :, delta:enc_feature.size()[2]-delta,
                                        delta:enc_feature.size()[2]-delta]
        return enc_feature

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        enc_fmap = self.encoder(x)
        x = self.decoder(enc_fmap[-1], enc_fmap[::-1][1:])
        x = self.final(x)
        print(x.shape)
        return x

net = UNet()
net(torch.rand((1, 1, 572, 572)))

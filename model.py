from torch import nn


class Flatten(nn.Module):

    def forward(self, x):
        _, C, H, W = x.shape
        return x.view(-1, C * H * W)


class MnistSoftmaxRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(Flatten(), nn.Linear(784, 10))

    def forward(self, x, return_probas=False):
        x = self.layers(x)
        return nn.functional.softmax(x, dim=-1) if return_probas else x


class MnistCNN(nn.Module):

    def __init__(
            self,
            pad=1,
            pool_stride=2,
            p_drop=0.15,
            n_classes=10,
            activation=nn.ReLU
    ):
        super().__init__()
        self.pad = 1
        self.pool_stride = pool_stride
        self.p_drop = p_drop
        self.n_classes = n_classes
        self.activation = nn.ReLU
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=pad),
            activation(),
            nn.MaxPool2d(2, stride=pool_stride),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=pad),
            activation(),
            nn.MaxPool2d(2, stride=pool_stride),
            nn.BatchNorm2d(32)
        )
        self.linear = nn.Sequential(
            Flatten(),
            nn.Linear(32 * 5 * 5, 256),
            activation(),
            nn.Dropout(p_drop),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            activation(),
            nn.Dropout(p_drop),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.convolution(x)
        return self.linear(x)

import chainer
import chainer.functions as F
import chainer.links as L

class ConvBlock(chainer.Chain):
    """A convolution, batch norm, ReLU block.
    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.
    For the convolution operation, a square filter size is used.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.
    """
    def __init__(self, in_channels, out_channels, ksize, pad=1, nobias=False):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize, pad=pad, nobias=nobias)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)

class LinearBlock(chainer.Chain):
    """A linear, batch norm, ReLU block.
    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.
    For the convolution operation, a square filter size is used.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(in_channels, out_channels)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.fc(x)
        h = self.bn(h)
        return F.relu(h)


class TNet(chainer.Chain):
    def __init__(self):
        super(TNet, self).__init__()
        with self.init_scope():
            self.conv1 = ConvBlock(3, 64)
            self.conv2 = ConvBlock(64, 128)
            self.conv3 = ConvBlock(128, 1024)
            self.fc1 = LinearBlock(1024, 512)
            self.fc2 = LinearBlock(512, 256)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pooling_2d(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class TNetFeat(chainer.Chain):
    def __init__(self):
        super(TNetFeat, self).__init__()
        with self.init_scope():
            self.conv1 = ConvBlock(64, 64)
            self.conv2 = ConvBlock(64, 128)
            self.conv3 = ConvBlock(128, 1024)
            self.fc1 = LinearBlock(1024, 512)
            self.fc2 = LinearBlock(512, 256)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pooling_2d(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PointNetCls(chainer.Chain):
    def __init__(self):
        super(PointNetCls, self).__init__()
        with self.init_scope():
            self.tnet = TNet()
            self.conv1_1 = ConvBlock(3, 64)
            self.conv1_2 = ConvBlock(64, 64)
            self.tnet = TNetFeat()
            self.conv2_1 = ConvBlock(64, 64)
            self.conv2_2 = ConvBlock(64, 128)
            self.conv2_3 = ConvBlock(128, 1024)
            self.fc1 = LinearBlock(None, 512)
            self.fc2 = LinearBlock(None, 512)
            self.fc3 = LinearBlock(None, 512)
t
    def __call__(self, x):
        pass

def loss(x, t):
    return 0

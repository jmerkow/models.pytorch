import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models_pytorch.utils import get_activation


# Reference: Modified from: (1) https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# (2) https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py

# ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    """
    Conv-LSTM Cell based on "Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting", arXiv: https://arxiv.org/pdf/1506.04214.pdf
    """

    def __init__(self, feature_size, in_planes, hidden_planes, kernel_size, bias=True):
        """
        feature_size: (int, int)
            (height, width) of input feature (tensor)
        in_planes: int
            Number of channels in input feature (tensor)
        hidden_planes: int
            Number of channels in hidden state (tensor)
        kernel_size: int
            Size of the convolutional kernel
        bias: bool
            Whether to add the bias or not
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = feature_size
        self.in_planes = in_planes
        self.hidden_planes = hidden_planes
        self.kernel = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Equation: 3 from paper
        # Wxi * Xt
        self.WiXconv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Whi * Ht-1
        self.WiHconv = nn.Conv2d(in_channels=self.hidden_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Wxf * Xt
        self.WfXconv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Whf * Ht-1
        self.WfHconv = nn.Conv2d(in_channels=self.hidden_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Wxc * Xt
        self.WcXconv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Whc * Ht-1
        self.WcHconv = nn.Conv2d(in_channels=self.hidden_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Wxo * Xt
        self.WoXconv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        # Who * Ht-1
        self.WoHconv = nn.Conv2d(in_channels=self.hidden_planes, out_channels=self.hidden_planes,
                                 kernel_size=self.kernel, stride=1, padding=self.padding, bias=self.bias)

        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_planes, self.height, self.width))
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_planes, self.height, self.width))
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_planes, self.height, self.width))

    def forward(self, x, h_prev, c_prev):
        """
        x: Tensor from CNN
        h_prev: Output from previous step
        c_prev: Output from previous step
        """
        # Ignoring bias in the equations below
        it = torch.sigmoid(self.WiXconv(x) + self.WiHconv(h_prev) + c_prev * self.Wci)
        ft = torch.sigmoid(self.WfXconv(x) + self.WfHconv(h_prev) + c_prev * self.Wcf)
        ct = ft * c_prev + it * torch.tanh(self.WcXconv(x) + self.WcHconv(h_prev))
        ot = torch.sigmoid(self.WoXconv(x) + self.WoHconv(h_prev) + ct * self.Wco)
        ht = ot * torch.tanh(ct)
        return ht, ct

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_planes, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_planes, self.height, self.width)).cuda())


# ConvLSTM
class ConvLSTM(nn.Module):
    """
    ConvLSTM based on "Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting", arXiv: https://arxiv.org/pdf/1506.04214.pdf
    """

    def __init__(self, feature_size, in_planes, hidden_planes, kernel_size, bias=True):
        """
        feature_size: (int, int)
            (height, width) of input feature (tensor)
        in_planes: int
            Number of channels in input feature (tensor)
        hidden_planes: list of int
            List of number of channels in hidden state (tensor)
        kernel_size: int
            Size of the convolutional kernel
        bias: bool
            Whether to add the bias or not
        """
        super(ConvLSTM, self).__init__()
        self.height, self.width = feature_size
        self.in_planes = [in_planes] + hidden_planes
        self.hidden_planes = hidden_planes
        self.kernel_size = kernel_size
        self.bias = bias
        self.num_cells = len(hidden_planes)
        cell_list = []
        for i in range(self.num_cells):
            cell = ConvLSTMCell(feature_size=(self.height, self.width),
                                in_planes=self.in_planes[i],
                                hidden_planes=self.hidden_planes[i],
                                kernel_size=self.kernel_size, bias=self.bias)
            cell_list.append(cell)
        self.cells = nn.ModuleList(cell_list)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_cells):
            init_states.append(self.cells[i].init_hidden(batch_size))
        return init_states

    def forward(self, feature, hidden_state=None):
        """
        feature: 5-D Tensor of shape (BS, T, C, H, W)
        """
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=feature.size(0))

        # Sequence length
        T = feature.size(1)
        curr_input = feature

        cells_output = []
        last_state_list = []
        for cell_idx in range(self.num_cells):
            h, c = hidden_state[cell_idx]
            # Loop through sequence
            seq_output = []
            for t in range(T):
                h, c = self.cells[cell_idx](x=curr_input[:, t, :, :, :],
                                            h_prev=h, c_prev=c)
                seq_output.append(h)

            cell_output = torch.stack(seq_output, dim=1)
            curr_input = cell_output

            cells_output.append(cell_output)
            last_state_list.append([h, c])

        return cells_output[-1:], last_state_list[-1:]


class FullyConnected(nn.Module):
    def __init__(self, in_features=64, nclasses=6, use_relu6=True):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.nclasses = nclasses
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.in_features, self.in_features * 2)
        if use_relu6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.in_features * 2, self.nclasses)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvLSTMClassifier(nn.Module):

    def __init__(self, encoder_channels, activation='sigmoid',
                 nclasses=6,
                 feature_shape=(7, 7),
                 hidden_planes=(128, 64, 64), kernel_size=5, bias=True,
                 encoder_index=0,
                 fc_relu6=True,
                 ):
        self.encoder_index = encoder_index
        self.activation_type = str(activation)
        self.hidden_planes = list(hidden_planes)
        super(ConvLSTMClassifier, self).__init__()
        self.activation = get_activation(activation)

        self.is_multi_task = False

        self.nclasses = nclasses
        self.feature_shape = tuple(feature_shape)

        self.conv_lstm = ConvLSTM(feature_shape, encoder_channels[self.encoder_index], self.hidden_planes,
                                  kernel_size, bias=bias)

        self.fc = FullyConnected(in_features=self.hidden_planes[-1], nclasses=self.nclasses,
                                 use_relu6=fc_relu6)

    def output_info(self):
        return {'final': {'nclasses': self.nclasses, 'activation': self.activation_type}}

    def forward(self, features, **kwargs):
        features = features[self.encoder_index]
        features = F.interpolate(features, size=self.feature_shape).unsqueeze(0)
        cells_output, states_output = self.conv_lstm(features)
        return self.fc(cells_output[0])

    def predict(self, features, **kwargs):
        return self.activation(self.forward(features, **kwargs))

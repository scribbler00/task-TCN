__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"

from torch import nn
import torch
from dies.abstracts import Transfer


class LSTMModel(Transfer):
    # based on https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
    def __init__(
        self,
        timeseries_length,
        n_features_hidden_state,
        num_recurrent_layer,
        output_dim,
        dropout=0.0,
    ):
        """

        Parameters
        ----------
        timeseries_length : integer
            the number of expected features in the input x.
        n_features_hidden_state : integer
            the number of features in the hidden state h.
        num_recurrent_layer : integer
            amount of recurrent layers, where the first takes the input, and the last yields the final output.
        output_dim : integer
            the desired dimension of the final output.
        dropout : float
            percentage used for the dropout layer.
        """
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = n_features_hidden_state

        # Number of hidden layers
        self.layer_dim = num_recurrent_layer

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=timeseries_length,
            hidden_size=n_features_hidden_state,
            num_layers=num_recurrent_layer,
            batch_first=True,
            dropout=0,
        )

        # Readout layer
        self.fc = nn.Linear(n_features_hidden_state, output_dim)

    def forward(self, categorical_data, continuous_data):
        """

        Parameters
        ----------
        x : pytorch.Tensor
            the given input time series.

        Returns
        -------
        pytorch.Tensor
            the output of the lstm network, fed into a linear layer to achieve the desired dimension.
        """
        x = continuous_data.type(torch.FloatTensor)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # increase dimension of output to match target dimension
        out = out.unsqueeze(1)
        # out.size() --> 100, 10
        return out

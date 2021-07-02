from itertools import chain
from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .mlp import Embedding


from .utils_pytorch import init_net
from dies.mlp import MultiLayerPerceptron
from dies.abstracts import Transfer
from dies.utils_pytorch import freeze, unfreeze

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class Autoencoder(Transfer):
    def __init__(
        self,
        ann_structure,
        hidden_activation=nn.LeakyReLU(negative_slope=0.1),
        use_batch_norm=True,
        final_layer_activation=None,
        dropout=None,
        embedding_module=None,
        embeding_position="start",
        y_ranges=None,
    ):
        """Class to create an autoencoder network, based on fully connected neural networks
        Args:
            ann_structure (list of ints): the following network structure
            hidden_activation (torch.nn.Module, optional): Activation function used after each hidden layer. Defaults to nn.LeakyReLU(negative_slope=0.1).
            use_batch_norm (bool, optional): Select if batchnorm layer are included after each hidden layer. Defaults to True.
            final_layer_activation (torch.nn.Module, optional): Activation funciton used after the final layer. Defaults to None.
            dropout (float, optional): dropout probability after each hidden layer. Defaults to None.
            embedding_module (dies.embedding.Embedding, optional): Embedding module for categorical features. Defaults to None.
            embeding_position (str, optional): Position at which the output of the embedding layer is included, Either "start" or "bottleneck". Defaults to "start".
            y_ranges ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
            Warning: [description]
        """

        super().__init__()

        ann_structure = copy.copy(ann_structure)

        self.latent_size = ann_structure[-1]
        self.input_size = ann_structure[0]
        self.output_size = self.input_size

        # save number of potential y_ranges
        self.y_ranges = y_ranges

        self.embedding_module = embedding_module
        self.embeding_position = embeding_position

        if (
            self.embedding_module is not None
            and self.embeding_position != "start"
            and self.embeding_position != "bottleneck"
        ):
            raise ValueError("embeding_position must be start or bottleneck")

        if self.embedding_module is not None and self.embeding_position == "start":
            self.input_size = self.input_size + self.embedding_module.no_of_embeddings
            ann_structure[0] = self.input_size

        self.network_structure_encoder = ann_structure.copy()
        self.network_structure_decoder = [ls for ls in reversed(ann_structure)]
        # replace finaly layer due to embedding module
        self.network_structure_decoder[-1] = self.output_size

        self.encoder_size = self.network_structure_encoder.copy()
        self.decoder_size = self.network_structure_decoder.copy()

        if self.network_structure_encoder[-1] != self.network_structure_decoder[0]:
            raise Warning(
                "Last element of encoder should match first element of decoder."
                + "But is {} and {}".format(
                    self.network_structure_encoder, self.network_structure_decoder
                )
            )

        if self.embedding_module is not None and self.embeding_position == "bottleneck":
            self.network_structure_decoder[0] = (
                self.network_structure_decoder[0]
                + self.embedding_module.no_of_embeddings
            )

        self.network_structure_encoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_encoder[0:-1], self.network_structure_encoder[1:]
            )
        ]
        self.network_structure_decoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_decoder[0:-1], self.network_structure_decoder[1:]
            )
        ]

        self.encoder = init_net(
            self.network_structure_encoder,
            hidden_activation,
            use_batch_norm,
            dropout=dropout,
            combine_to_sequential=True,
        )
        self.decoder = init_net(
            self.network_structure_decoder,
            hidden_activation,
            use_batch_norm,
            include_activation_final_layer=False,
            dropout=dropout,
            combine_to_sequential=True,
        )

        if final_layer_activation is not None:
            self.decoder.append(final_layer_activation)

        self.encoder_ = nn.Sequential(*self.encoder)
        self.decoder_ = nn.Sequential(*self.decoder)

        self.n_gpu = torch.cuda.device_count()

    def _encode(
        self,
        continuous_data,
        categorical_data,
    ):

        if (
            self.embeding_position == "start"
            and (categorical_data is not None or len(categorical_data) > 0)
            and self.embedding_module is not None
        ):
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, continuous_data], 1)
        else:
            x = continuous_data

        x = self.encoder_(x)

        if (
            self.embeding_position == "bottleneck"
            and categorical_data is not None
            and self.embedding_module is not None
        ):

            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, x], 1)

        return x

    def forward(self, categorical_data, continuous_data):

        x = self._encode(
            continuous_data=continuous_data, categorical_data=categorical_data
        )

        x = self.decoder_(x)

        if self.y_ranges is not None:
            for idx, y_range in enumerate(self.y_ranges):
                x[:, idx] = (y_range[1] - y_range[0]) * torch.sigmoid(
                    x[:, idx]
                ) + y_range[0]

        return x

    def freeze_to(self, n, include_embedding=True):
        self.unfreeze()
        if n < 0:
            n = len(self.layers) + n

        for idx, l in enumerate(self.encoder_):
            if idx < n:
                freeze(l)
            else:
                unfreeze(l)

        for idx, l in enumerate(self.decoder_):
            if (idx + len(self.encoder_)) < n:
                freeze(l)
            else:
                unfreeze(l)

        if self.embedding_module is not None:
            for emb in self.embedding_module.embeddings:
                if include_embedding:
                    freeze(emb)
                else:
                    unfreeze(emb)


class VariationalAutoencoder(Autoencoder):
    def __init__(
        self,
        ann_structure,
        hidden_activation=nn.LeakyReLU(negative_slope=0.1),
        use_batch_norm=True,
        final_layer_activation=None,
        dropout=None,
        embedding_module=None,
        embeding_position="start",
        y_ranges=None,
    ):
        """
        input_size:
        ann_structure:
        hidden_activation:
        use_batch_norm:
        dropout: {float} Amount of dropout to be used.
        final_layer_activation: Activation function to be used in the final layer.
        """
        super(VariationalAutoencoder, self).__init__(
            ann_structure,
            hidden_activation=hidden_activation,
            use_batch_norm=use_batch_norm,
            final_layer_activation=final_layer_activation,
            dropout=dropout,
            embedding_module=embedding_module,
            embeding_position="start",
            y_ranges=y_ranges,
        )
        self.representation_size = ann_structure[-1]
        self.en_mu = nn.Linear(ann_structure[-1], self.representation_size)
        self.en_std = nn.Linear(ann_structure[-1], self.representation_size)

        self.de1 = nn.Linear(self.representation_size, ann_structure[-1])

    def _encode(self, continuous_data, categorical_data=None):
        """Encode a batch of samples, and return posterior parameters for each point."""

        if (
            self.embeding_position == "start"
            and categorical_data is not None
            and self.embedding_module is not None
        ):
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, continuous_data], 1)
            self.last_cat = categorical_data
        elif (
            self.embeding_position == "start"
            and categorical_data is None
            and self.embedding_module is not None
        ):
            x = torch.cat([self.last_cat, continuous_data], 1)
        else:
            x = continuous_data

        #         h1 = self.act_func(self.en1(x))
        h1 = self.encoder_(x)

        return self.en_mu(h1), self.en_std(h1)

    def decode(self, z):
        """Decode a batch of latent variables"""
        h2 = self.de1(z)
        return self.decoder_(h2)

    def forward(self, categorical_data, continuous_data):

        mu, logvar = self._encode(continuous_data, categorical_data)

        z = self.reparam(mu, logvar)
        x = self.decoder_(z)

        if self.y_ranges is not None:
            for idx, y_range in enumerate(self.y_ranges):
                x[:, idx] = (y_range[1] - y_range[0]) * torch.sigmoid(
                    x[:, idx]
                ) + y_range[0]
        return x

    def get_posteriors(self, continuous_data, categorical_data=None):

        return self._encode(continuous_data, categorical_data)

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""

        mu, logvar = self.encode(x)
        return self.reparam(mu, logvar)

    def reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values.
        This is stochastic during training,  and returns the mode during evaluation."""

        if self.training:
            # convert logarithmic variance to standard deviation representation
            std = logvar.mul(0.5).exp_()
            # create normal distribution as large as the data
            eps = Variable(std.data.new(std.size()).normal_())
            # scale by learned mean and standard deviation
            return eps.mul(std).add_(mu)
        else:
            return mu

    def freeze_to(self, n, include_embedding=True):
        self.unfreeze()
        if n < 0:
            n = len(self.layers) + n

        for idx, l in enumerate(self.encoder_):
            if idx < n:
                freeze(l)
            else:
                unfreeze(l)

        if idx + 1 < n:
            freeze(self.en_mu)
            freeze(self.en_std)
        else:
            unfreeze(self.en_mu)
            unfreeze(self.en_std)

        if idx + 2 < n:
            freeze(self.de1)
        else:
            unfreeze(self.de1)

        for idx, l in enumerate(self.decoder_):
            if (idx + len(self.encoder_) + 2) < n:
                freeze(l)
            else:
                unfreeze(l)

        if self.embedding_module is not None:
            for emb in self.embedding_module.embeddings:
                if include_embedding:
                    freeze(emb)
                else:
                    unfreeze(emb)


class ConvolutionalAutoEncoder(Autoencoder):
    def __init__(
        self,
        ann_structure,
        kernel_size,
        padding=False,
        hidden_activation=nn.LeakyReLU(negative_slope=0.1),
        use_batch_norm=True,
        final_layer_activation=None,
        dropout=None,
        embedding_module=None,
        embeding_position="start",
        y_ranges=None,
        debug=False,
    ):
        """[summary]

        Arguments:
            cnn_structure {Array} -- Structure of the encoder side of the CNN, e.g. [16,12,8,4]
            kernel_size {int} -- [description]

        Keyword Arguments:
            adjust_groups {bool} -- [description] (default: {False})
            debug {bool} -- [description] (default: {False})
        """

        super(ConvolutionalAutoEncoder, self).__init__(
            ann_structure,
            hidden_activation=hidden_activation,
            use_batch_norm=use_batch_norm,
            final_layer_activation=final_layer_activation,
            dropout=dropout,
            embedding_module=embedding_module,
            embeding_position="start",
            y_ranges=y_ranges,
        )
        self.debug = debug

        self.padding = kernel_size // 2 if padding else 0

        self.encoder_ = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"cnn_layer_{idx}_{input_size}_{output_size}",
                        nn.Conv1d(
                            input_size, output_size, kernel_size, padding=self.padding
                        ),
                    )
                    for idx, (input_size, output_size) in enumerate(
                        zip(self.encoder_size[0:-1], self.encoder_size[1:])
                    )
                ]
            )
        )
        self.decoder_ = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"deconv_layer_{idx}_{input_size}_{output_size}",
                        nn.ConvTranspose1d(
                            input_size, output_size, kernel_size, padding=self.padding
                        ),
                    )
                    for idx, (input_size, output_size) in enumerate(
                        zip(self.decoder_size[0:-1], self.decoder_size[1:])
                    )
                ]
            )
        )
        self.main = nn.Sequential(self.encoder_, self.decoder_)

    def get_encoder(self, continuous_data, categorical_data=None):
        return self._encode(continuous_data, categorical_data)

    def shape_print(self, name, output):
        if self.debug:
            print("{}: {}".format(name, output.shape))

    def _encode(
        self,
        categorical_data,
        continuous_data,
    ):

        if self.embeding_position == "start" and self.embedding_module is not None:
            features = categorical_data.shape[1]
            timesteps = categorical_data.shape[2]
            batch_size = categorical_data.shape[0]

            categorical_data = self.embedding_module(
                categorical_data.permute(0, 2, 1).reshape(-1, features)
            )

            x = torch.cat(
                [
                    categorical_data.reshape(
                        -1, timesteps, categorical_data.shape[1]
                    ).permute(0, 2, 1),
                    continuous_data,
                ],
                1,
            )

        else:
            x = continuous_data

        x = self.encoder_(x)

        if self.embeding_position == "bottleneck" and self.embedding_module is not None:
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, x], 1)

        return x


class AEMLP(torch.nn.Module):
    def __init__(self, autoencoder, mlp):
        """Combines an autoencoder and and MLP.
            First encodes the input data and then forwards it to an MLP to create the final output.

        Args:
            autoencoder (Autoencoder): The autoencoder to encode the input data.
            mlp (MultiLayerPerceptron): The MLP to create the response.
        """
        super(AEMLP, self).__init__()
        self.autoencoder = autoencoder
        self.mlp = mlp

        if not isinstance(self.autoencoder, Autoencoder):
            raise TypeError
        if not isinstance(self.mlp, MultiLayerPerceptron):
            raise TypeError

    def forward(self, categorical_data, continuous_data):
        x = self.autoencoder._encode(continuous_data, categorical_data)
        x = self.mlp(x)

        return x

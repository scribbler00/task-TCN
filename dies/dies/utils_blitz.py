from blitz.modules import BayesianLinear
from blitz.modules import BayesianEmbedding, BayesianConv1d
from blitz.modules.base_bayesian_module import BayesianModule
from torch import nn
import torch


def convert_layer_to_bayesian(layer):
    if isinstance(layer, torch.nn.Linear):
        new_layer = BayesianLinear(layer.in_features, layer.out_features)
    elif isinstance(layer, nn.Embedding):
        new_layer = BayesianEmbedding(layer.num_embeddings, layer.embedding_dim)
    elif isinstance(layer, nn.Conv1d):
        new_layer = BayesianConv1d(
            layer.in_channels,
            layer.out_channels,
            kernel_size=layer.kernel_size[0],
            groups=layer.groups,
            padding=layer.padding,
            dilation=layer.dilation,
        )
    else:
        Warning(
            f"Could not find correct type for conversion of layer {layer} with type {type(layer)}"
        )
        new_layer = layer

    return new_layer


def convert_to_bayesian_model(model):
    for p in model.named_children():
        cur_layer_name = p[0]
        cur_layer = p[1]
        if len(list(cur_layer.named_children())) > 0:
            convert_to_bayesian_model(cur_layer)
        elif not isinstance(cur_layer, BayesianModule):
            new_layer = convert_layer_to_bayesian(cur_layer)
            setattr(model, cur_layer_name, new_layer)

    return model


def set_train_mode(model, mode):
    if isinstance(model, BayesianModule):
        model.freeze = not mode

    for module in model.children():
        set_train_mode(module, mode)
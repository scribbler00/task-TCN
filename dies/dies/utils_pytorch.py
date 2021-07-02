__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pickle as pickle
import time
from inspect import signature
from torch.nn import Embedding
import inspect
from .utils import np_int_dtypes
import copy
import warnings


def np_to_dev(xx, device=None):
    """
    Transform a list of given values into a Tensor, and attach it to a specific device.

    Parameters
    ----------
    xx : pytorch.Tensor/list/pandas.DataFrame
        values that are to be transformed into a Tensor.
    device : pytorch.device
        determines which device the calculations are to be performed on.

    Returns
    -------
    pytorch.Tensor
        input values as a tensor, attached to the given device.
    """
    xx = np.array(xx)

    if xx.dtype in np_int_dtypes:
        dtype = torch.LongTensor
    else:
        dtype = torch.FloatTensor

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(xx, np.ndarray):
        xx = torch.from_numpy(xx)

    return xx.type(dtype).to(device)


def dev_to_np(xx):
    """
    Transform a Tensor into an array.

    Parameters
    ----------
    xx : pytorch.Tensor/list of pytorch.Tensors
        values that are to be transformed into an array.

    Returns
    -------
    numpy.array
        input values as an array.
    """
    if isinstance(xx, np.ndarray):
        return xx
    if type(xx) == tuple:
        return tuple([dev_to_np(xx_i) for xx_i in xx])
    if type(xx) == list:
        return [dev_to_np(xx_i) for xx_i in xx]

    if xx.is_cuda:
        xx = xx.cpu()
    return xx.detach().numpy()


def df_to_dev(xx, device=None):
    """
    Transform a DataFrame into a Tensor, and attach it to a specific device.

    Parameters
    ----------
    xx : pandas.DataFrame
        values that are to be transformed into a Tensor.
    device : pytorch.device
        determines which device the calculations are to be performed on.

    Returns
    -------
    pytorch.Tensor
        input values as a tensor, attached to the given device.
    """
    return np_to_dev(xx.values, device)


def kaiming_init_normal(m):
    """
    Set the values of a pytorch layer according to the Kaiming normal initialization/He initialization.

    Parameters
    ----------
    m : any pytorch.Conv layer/pytorch.Linear
        the pytorch layer whose weights are to be set.

    Returns
    -------

    """
    if _is_correct_type(m):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.1)


def kaiming_init_uniform(m):
    """
    Set the values of a pytorch layer according to the Kaiming uniform initialization/He initialization.

    Parameters
    ----------
    m : any pytorch.Conv layer/pytorch.Linear
        the pytorch layer whose weights are to be set.

    Returns
    -------

    """
    if _is_correct_type(m):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.1)


def xavier_init_normal(m):
    """
    Set the values of a pytorch layer according to the Xavier normal initialization/Glorot initialization.

    Parameters
    ----------
    m : any pytorch.Conv layer/pytorch.Linear
        the pytorch layer whose weights are to be set.

    Returns
    -------

    """
    if _is_correct_type(m):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)


def xavier_init_uniform(m):
    """
    Set the values of a pytorch layer according to the Xavier uniform initialization/Glorot initialization.

    Parameters
    ----------
    m : any pytorch.Conv layer/pytorch.Linear
        the pytorch layer whose weights are to be set.

    Returns
    -------

    """
    if _is_correct_type(m):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


def _is_correct_type(m):
    """
    Determine whether the type of a given object is a pytorch convolution or pytorch linear layer.

    Parameters
    ----------
    m : any
        the object whose type is to be assured.

    Returns
    -------
    bool
        True if the type is correct, False otherwise.
    """
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Conv3d)
    ):
        return True
    else:
        return False


def initialize_weights(m, init_function=torch.nn.init.xavier_normal_, gain=1):
    """
    Set the weights of the given pytorch layer with the selected function, using the gain value if provided.

    Parameters
    ----------
    m : pytorch.Conv1d/pytorch.Linear
        the pytorch layer whose weights are to be set.
    init_function : pytorch.nn.init
        any function to initialize the weights of a layer.
    gain : integer/float
        the gain value for the chosen 'init_function'

    Returns
    -------

    Notes
    -----
    Make sure to calculate the gain yourself using
    https://pytorch.org/docs/master/nn.init.html#torch.nn.init.calculate_gain
    """
    if "gain" in signature(init_function).parameters:
        init_fn = lambda x: init_function(x, gain=gain)
    else:
        init_fn = lambda x: init_function(x)

    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)
        m.bias.data.fill_(0.1)
    if isinstance(m, torch.nn.Conv1d):
        init_fn(m.weight.data)
        m.bias.data.fill_(0.1)


def save_paramters_dict(name, parameters):
    """
    Store the provided parameters at a specified location.

    Parameters
    ----------
    name : string
        the name of the file in which the parameters are to be stored.
    parameters : pytorch.nn.parameter
        the parameters that are to be stored.
    Returns
    -------

    """
    with open(name + "_hyper_parameter.par", "wb") as f:
        pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)


def load_parameters_dict(name):
    """
    Load the parameters from a specified location.

    Parameters
    ----------
    name : string
        the name of the file in which the parameters have been stored.

    Returns
    -------
    kwargs : pytorch.nn.parameter
        the read parameters.
    """
    with open(name + "_hyper_parameter.par", "rb") as f:
        kwargs = pickle.load(f)

    return kwargs


def pickle_object(model, name):
    """
    Use pickle to store the given model at a specified location.

    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model that is to be stored.
    name : string
        the name of the file in which the parameters are to be stored.

    Returns
    -------

    """
    with open(name + ".pt", "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def save_model(model, name, include_time_stamp=False, **kwargs):
    """
    Stores a model and optionally adds a time stamp.

    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model that is to be stored.
    name : string
        the name of the file in which the parameters are to be stored.
    include_time_stamp : bool
        if True, add the current time to the output file name.
    kwargs : pytorch.nn.parameter
        the parameters that are to be stored.
    Returns
    -------

    """
    if include_time_stamp:
        name = "" + name + "_" + str(time.time())

    with open(name + ".par", "wb") as f:
        pickle.dump(kwargs, f, pickle.HIGHEST_PROTOCOL)

    torch.save(model.to("cpu").state_dict(), name + ".pt")


def load_model(
    ModelClass,
    path_to_model,
    device=None,
    include_missing_dict_items=False,
    strict=False,
    additional_dict_params=None,
    post_process_params=None,
    remove_additional_dict_items=False,
):
    """
    Loads a model from the given path. Model is returned in eval mode.

    Parameters
    ----------
    ModelClass : torch.nn
        the class of which an object with the given parameters is to be created.
    path_to_model : string
        the path of the file in which the model is stored.
    device : pytorch.device
        determines which device the calculations are to be performed on.
    include_missing_dict_items : bool
        if True, add all keys that are missing in the state dictionary by using the newly created model's values.
    strict : bool
        if True, loading an incomplete state dictionary/a state dictionary with too many keys yields an error.
    additional_dict_pararms : pytorch.nn.parameter
        if provided, add them to the loaded parameters.
    post_process_params : function
        if provided, apply it on the combined parameters.
    remove_additional_dict_items : bool
        if True, check whether the parameters can be used by the model, and remove those that can not.

    Returns
    -------

    """
    path_to_model = path_to_model

    with open(path_to_model + ".par", "rb") as f:
        kwargs = pickle.load(f)

    if additional_dict_params is not None:
        kwargs = {**kwargs, **additional_dict_params}
    if post_process_params is not None:
        kwargs = post_process_params(kwargs)

    if remove_additional_dict_items:
        valid_args = inspect.getargspec(ModelClass.__init__).args
        for k in list(kwargs.keys()):
            if k not in valid_args:
                del kwargs[k]

    model = ModelClass(**kwargs)

    if device is None:
        map_location = "cpu"
    else:
        map_location = None

    state_dict_pre_trained = torch.load(
        path_to_model + ".pt", map_location=map_location
    )

    if include_missing_dict_items:
        state_dict_missing_values = {
            k: v
            for k, v in model.state_dict().items()
            if k not in state_dict_pre_trained
        }
        state_dict_pre_trained.update(state_dict_missing_values)

    model.load_state_dict(state_dict_pre_trained, strict=strict)

    if device is not None:
        model.to(device)

    model.eval()

    return model


def init_net(
    network_structure,
    hidden_activation,
    use_batch_norm,
    include_activation_final_layer=True,
    dropout=None,
    combine_to_sequential=False,
):
    """

    Parameters
    ----------
    network_structure : list of integers
        the sizes of each layer, for which a subnetwork is created.
    hidden_activation : pytorch.nn
        activation function that is to be appended to every subnetwork.
    use_batch_norm : bool
        if True, add a normalization layer to every subnetwork.
    include_activation_final_layer : bool
        if True, also append an activation function to the last subnetwork.
    dropout : float
        if provided, add a dropout layer to each subnetwork.
    combine_to_sequential : bool
        if True, combine all subnetworks to a pytorch.nn.Sequential. Otherwise, combine them to a list.

    Returns
    -------
    list of lists/list of pytorch.nn.Sequential
        the initialized pytorch network.
    """
    net = []

    # assures compatibility between single tasks and multitasks
    if not isinstance(network_structure, list):
        network_structure = [network_structure]

    for idx, x in enumerate(network_structure):
        sub_net = [x]
        if idx != len(network_structure) - 1 or include_activation_final_layer:
            if hidden_activation is not None:
                sub_net.append(hidden_activation)
            # as batch norm aims to "control" the input to the next layer
            # it makes sense to do it after the activation to take the actual value
            # including the non-linear activation into account
            # see to https://blog.paperspace.com/busting-the-myths-about-batch-normalization
            if use_batch_norm:
                sub_net.append(nn.BatchNorm1d(x.out_features))
                # sub_net.append(RunningBatchNorm(x.out_features))
            # https://forums.fast.ai/t/questions-about-batch-normalization/230
            if dropout is not None:
                sub_net.append(nn.Dropout(dropout))

        if combine_to_sequential:
            net.append(nn.Sequential(*sub_net))
        else:
            net.append(sub_net)

    return net


def unfreeze_n_final_layer(model, n, include_embedding=False):
    """
    Remove all but the last 'n' layers from the gradient computation.

    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model whose layers are to be excluded from the gradient computation.
    n : interger
        the number of layers not to be included for gradient computation.
    include_embedding : bool
        if True, include all embedding layers to the gradient computation.

    Returns
    -------

    Notes
    -----
    Currently embedding layers are either included or excluded through 'include_embedding'.
    """
    # freeze all parameters by excluding them from gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Reinclude the parameters of the last n layers to gradient computation
    layers = list(model.children())

    new_layers = []
    for l in layers:
        if type(l) is nn.ModuleList:
            unfreeze_n_final_layer(l, n, include_embedding=include_embedding)
        elif type(l) is Embedding and include_embedding:
            for param in l.parameters():
                param.requires_grad = True
        elif type(l) is Embedding and not include_embedding:
            for param in l.parameters():
                param.requires_grad = False
        elif hasattr(l, "weight") or isinstance(l, nn.Sequential):
            new_layers.append(l)

    if len(new_layers) > 0:
        layers = new_layers

        if n > len(layers) or n == -1:
            n = len(layers)  # relearn the whole network

        for i in range(1, n + 1):
            for param in layers[-i].parameters():
                param.requires_grad = True


def freeze(layer):
    """
    Exclude a layer from the gradient computation.
    Parameters
    ----------
    layer : torch.nn
        the layer which is to be excluded from the gradient computation.

    Returns
    -------

    """
    for p in layer.parameters():
        p.requires_grad = False


def unfreeze(layer):
    """
    Include a layer to the gradient computation.
    Parameters
    ----------
    layer : torch.nn
        the layer which is to be included to the gradient computation.

    Returns
    -------

    """
    for p in layer.parameters():
        p.requires_grad = True


def print_requires_grad(
    model, include_embedding=True, type_name="", rec_level=0, tabs=""
):
    """
    Print which layers of the model are included in the gradient computation.
    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model that is to be analyzed.
    include_embedding : bool
        currently not used.
    type_name : string
        currently not used.
    rec_level : integer
        currently not used.
    tabs : string
        the amount of space before each print.

    Returns
    -------

    """
    layers = list(model.children())
    new_rec_level = rec_level + 1

    modules = model._modules
    if isinstance(model, nn.ModuleList):
        cur_type = "ModuleList"
    elif isinstance(model, nn.Sequential):
        cur_type = "Sequential"
    else:
        cur_type = ""
    for k, v in modules.items():
        if len(v._modules) > 0:
            print(f"{tabs}{cur_type} ({k}): (")
            new_tabs = tabs + "  "
            print_requires_grad(v, tabs=new_tabs)
            print(f"{tabs})")
        else:
            if hasattr(v, "weight"):
                print(f"{tabs}({v}) Requires grad: {v.weight.requires_grad}")
            else:
                print(f"{tabs}({v})")


def transform_layer(input, from_inst, to_inst, args={}, attrs={}):
    if isinstance(input, from_inst):
        for key in args.keys():
            arg = args[key]
            if isinstance(arg, str):
                if arg.startswith("."):
                    args[key] = getattr(input, arg[1:])

        output = to_inst(**args)

        for key in attrs.keys():
            attr = attrs[key]
            if isinstance(attr, str):
                if attr.startswith("."):
                    attrs[key] = getattr(input, attr[1:])

            setattr(output, key, attrs[key])
    else:
        output = input
    return output


def transform_model(
    input, from_inst, to_inst, args={}, attrs={}, inplace=True, _warn=True
):
    if inplace:
        output = input
        if _warn:
            warnings.warn(
                "\n * Caution : The Input Model is CHANGED because inplace=True.",
                Warning,
            )
    else:
        output = copy.deepcopy(input)

    if isinstance(output, from_inst):
        output = transform_layer(
            output, from_inst, to_inst, copy.deepcopy(args), copy.deepcopy(attrs)
        )
    else:
        for name, module in output.named_children():
            setattr(
                output,
                name,
                transform_model(
                    module,
                    from_inst,
                    to_inst,
                    copy.deepcopy(args),
                    copy.deepcopy(attrs),
                    _warn=False,
                ),
            )

    return output
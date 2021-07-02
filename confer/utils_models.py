from fastcore.foundation import L


from dies.embedding import EmbeddingModule
from dies.cnn import *
from dies.mlp import MultiLayerPerceptron
from dies.utils import get_structure
from dies.losses import *


def get_cnn_model(
    n_input_features,
    config,
    categorical_dimensions,
    embedding_type,
    cnn_type="cnn",
    input_sequence_length=None,
    output_sequence_length=None,
):
    ann_structure = [n_input_features] + get_structure(
        n_input_features * config["size_multiplier"],
        config["percental_reduce"],
        11,
        final_outputs=[5, 1],
    )

    emb_at_ith_layer = config["emb_at_ith_layer"]

    embedding_module = EmbeddingModule(
        categorical_dimensions=categorical_dimensions,
        embedding_dropout=config["embedding_dropout"],
        embedding_type=embedding_type,
    )

    if emb_at_ith_layer == "first":
        combined_embds_at_layers = L(0)
    elif emb_at_ith_layer == "last":
        combined_embds_at_layers = L(len(ann_structure) - 2)
    elif emb_at_ith_layer == "first_and_last":
        combined_embds_at_layers = L(0, len(ann_structure) - 2)
    elif emb_at_ith_layer == "all":
        combined_embds_at_layers = L(range(len(ann_structure) - 2))
    else:
        combined_embds_at_layers = []
        embedding_module = None

    model = TemporalCNN(
        ann_structure,
        cnn_type=cnn_type,
        embedding_module=embedding_module,
        dropout=config["dropout"],
        add_embedding_at_layer=combined_embds_at_layers,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        act_func=nn.ReLU,
    )

    return model


def get_mlp_model(
    n_input_features,
    config,
    categorical_dimensions,
    embedding_type,
):
    ann_structure = [n_input_features] + get_structure(
        n_input_features * config["size_multiplier"],
        config["percental_reduce"],
        11,
        final_outputs=[5, 1],
    )

    embedding_module = EmbeddingModule(
        categorical_dimensions=categorical_dimensions,
        embedding_dropout=config["embedding_dropout"],
        embedding_type=embedding_type,
    )

    model = MultiLayerPerceptron(
        ann_structure,
        embedding_module=embedding_module,
        ps=config["dropout"],
        embed_p=config["embedding_dropout"],
    )

    return model

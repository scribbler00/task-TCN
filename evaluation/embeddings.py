# %%

from fastai.learner import load_learner
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# %%
source_model_file = "/home/scribbler/workspace/convolutional_transfer/results/2015_wind/tcn_source_model_True_0_EmbeddingType.Normal_residuals_all.pkl"
learner = load_learner(source_model_file)
num_timestamps = 96
task_id = 7
temporal_block_id = 5

embedding_space = learner.model._forward_embedding_module_same(
    torch.ones([1, 1, num_timestamps]).long() * task_id
)
embedding_transform = learner.model.layers.temporal_blocks[
    temporal_block_id
].embedding_transform

transformed_embedding = embedding_transform(
    embedding_space
)  # .reshape(-1, num_timestamps)
# %%
transformed_embedding = transformed_embedding.detach().numpy()
embedding_space = embedding_space  # .reshape(-1, num_timestamps).detach().numpy()
weights_transform = embedding_transform.weight
weights_transform = weights_transform  # .reshape(weights_transform.shape[0],weights_transform.shape[1]).detach().cpu()
# %%
sns.heatmap(transformed_embedding)
plt.title("Transformed embedding of a task.")
plt.ylabel("Transformed embedding.")
plt.xlabel("Time steps.")
plt.show()

sns.heatmap(embedding_space)
plt.title("Original learned embedding of a task.")
plt.ylabel("Learned embedding.")
plt.xlabel("Time steps.")
plt.show()

sns.heatmap(weights_transform)
plt.title("Transform embedding weights.")
# plt.ylabel("Learned embedding.")
# plt.xlabel("Time steps.")
plt.show()

sns.heatmap(weights_transform)
plt.title("Transform embedding weights.")
# plt.ylabel("Learned embedding.")
# plt.xlabel("Time steps.")
plt.show()

# %%

import pdb

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from toolz import partition_all, dissoc

st.set_page_config(layout="wide")

PATH_ATTENTIONS_H5 = "output/attentions-Controlled_Images_B.h5"
h5py_file = h5py.File(PATH_ATTENTIONS_H5, "r")

with st.sidebar:
    idx = st.number_input("Sample index", min_value=0, max_value=9, value=0)

    def get_key_str(result, key):
        return result[key][()].decode("utf-8")

    results = h5py_file[str(idx)]
    image = results["image"][()]
    question = get_key_str(results, "question")
    answer_pred = get_key_str(results, "generated-text")
    answer_true = get_key_str(results, "preposition")

    # cols = st.columns([1, 2])
    st.image(image)
    st.markdown(
        """
    - Question: {}
    - Answer (pred): {}
    - Answer (true): {}
                """.format(
            question,
            answer_pred,
            answer_true,
        )
    )

tokens = [t.decode("utf-8") for t in results["input-tokens"][()]]
attentions = results["attentions"][()]
num_layers, num_heads, seq_len = attentions.shape


def show_attention_to_image(attentions, layer_idx, head_idx):
    W, H = image.size
    PATCH_SIZE = 32
    W = W // PATCH_SIZE
    H = H // PATCH_SIZE
    IMAGE_TOKEN = "<|image_pad|>"
    image_idxs = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN]
    attentions = attentions[layer_idx, head_idx]
    attentions = attentions[image_idxs]
    attentions = attentions.reshape(H, W)
    max_val = attentions.max()
    attentions = attentions / max_val
    st.image(
        attentions,
        caption="L: {} · H: {} · max: {:.2f}".format(layer_idx, head_idx, max_val),
        # caption="{} {}".format(min(attentions.flatten()), max(attentions.flatten())),
        # use_column_width=True,
        use_container_width=True,
    )


st.markdown("### Attention to image tokens -- per layer and per head")
col0, col1, col2, *_ = st.columns([1, 1, 1, 3])
layers_start = col0.number_input(
    "First layer",
    min_value=0,
    max_value=num_layers - 1,
    value=0,
)
layers_step = col1.number_input(
    "Step",
    min_value=1,
    max_value=num_layers,
    value=5,
)
layers_end = col2.number_input(
    "Last layer",
    min_value=layers_start,
    max_value=num_layers,
    value=num_layers,
)


layer_idxs = list(range(layers_start, layers_end, layers_step))

for head_idx in range(num_heads):
    cols = st.columns(len(layer_idxs))
    for col, layer_idx in zip(cols, layer_idxs):
        with col:
            show_attention_to_image(attentions, layer_idx, head_idx)


st.markdown("### Attention to other tokens")
to_drop_start_tokens = st.checkbox("Ignore tokens before the question", value=False)


def show_attention_to_other(attentions, head_idx):
    IMAGE_TOKEN = "<|image_pad|>"
    other_idxs = [i for i, t in enumerate(tokens) if t != IMAGE_TOKEN]
    if to_drop_start_tokens:
        other_idxs = other_idxs[5:]
    other_tokens = [repr(tokens[i]) for i in other_idxs]
    attentions = attentions[:, head_idx]
    attentions = attentions[:, other_idxs]
    attentions = attentions.T  # (num_other_tokens, num_layers)
    _, num_layers = attentions.shape
    layers = ["{}".format(i) for i in range(num_layers)]
    df = pd.DataFrame(
        attentions,
        index=other_tokens,
        columns=layers,
    )
    S = 0.3
    ncols = attentions.shape[0]
    nrows = attentions.shape[1]
    fig, ax = plt.subplots(figsize=(S * ncols, S * nrows))
    sns.heatmap(df, square=True, cbar=False, ax=ax)
    ax.set_title("Head: {} · max: {:.2f}".format(head_idx, attentions.max()))
    ax.set_xlabel("Layer")
    fig.tight_layout()
    st.pyplot(fig)


ncols = 2
for group in partition_all(ncols, range(num_heads)):
    cols = st.columns(ncols)
    for col, head_idx in zip(cols, group):
        with col:
            show_attention_to_other(attentions, head_idx)

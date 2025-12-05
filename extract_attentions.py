import pdb
import re

import h5py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

from dataset_zoo import get_dataset
from scripts.inference_hf import create_question, set_seeds


set_seeds(42)

DATA_DIR = "data/whatsup_vlms_data"
DEVICE = "cuda"
PATH_ATTENTIONS_H5 = "output/attentions-{}.h5"

dataset_name = "Controlled_Images_B"
dataset = get_dataset(
    dataset_name,
    image_preprocess=None,
    download=False,
    root_dir=DATA_DIR,
)

PREPOSITION_MAPPING = {
    "right_of": "right",
    "left_of": "left",
    "in-front_of": "front",
    "behind": "behind",
}


prepositions = dataset.all_prepositions
prepositions = [PREPOSITION_MAPPING[p] for p in prepositions]
prepositions_set = set(prepositions)

# num_to_show = 16
# for group in partition_all(4, range(num_to_show)):
#     cols = st.columns(4)
#     for i, col in zip(group, cols):
#         datum = dataset[i]
#         col.image(datum["image_options"][0])
#         col.text(datum["caption_options"][0])


def get_model_and_processor():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    return model, processor


def get_datum(i):
    def extract_objects(caption):
        match = re.search(
            r"A (.+?) (?:to the (?:left|right) of|in front of|behind|on|under|above) a (.+?)$",
            caption,
        )
        return match.group(1), match.group(2)

    datum = dataset[i]
    image = datum["image_options"][0]
    caption = datum["caption_options"][0]
    preposition = prepositions[i]

    objects = extract_objects(caption)
    is_one_object = False

    assert not "one" in dataset_name.lower()

    question = create_question(objects, prepositions_set, is_one_object)

    return {
        "question": question,
        "caption": caption,
        "image": image,
        "objects": objects,
        "preposition": preposition,
    }


def prepare_inputs(processor, i):
    datum = get_datum(i)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": datum["image"],
                },
                {"type": "text", "text": datum["question"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(text)
    inputs = processor(
        text=[text],
        images=[datum["image"]],
        padding=True,
        return_tensors="pt",
    )
    inputs.to(DEVICE)
    return inputs


def extract_attentions():
    model, processor = get_model_and_processor()

    def get_attention_matrix(outputs):
        token_idx = 0
        batch_idx = 0
        attentions = outputs.attentions[token_idx]
        attentions = [attention_layer[batch_idx] for attention_layer in attentions]
        attentions = [attention.detach() for attention in attentions]
        attentions = torch.stack(attentions)
        # (num_layers, num_heads, seq_len, seq_len)
        return attentions

    def get_attention_rollout(attn, head_pooling="avg", add_residual=True):
        dim_head = 1
        POOL_FUNCS = {
            "avg": lambda x: x.mean(dim=dim_head),
            "max": lambda x: x.max(dim=dim_head).values,
            "min": lambda x: x.min(dim=dim_head).values,
        }
        num_layers, _, seq_len, _ = attn.shape

        # 1. Average heads per layer → (num_layers, seq_len, seq_len)
        attn_avg = POOL_FUNCS[head_pooling](attn)

        # 2. Add residual connection: A' = (A + I) / 2
        if add_residual:
            I = torch.eye(seq_len, device=attn.device)
            attn_res = (attn_avg + I) / 2.0
        else:
            attn_res = attn_avg

        # 3. Row-normalize
        attn_norm = attn_res / attn_res.sum(dim=-1, keepdim=True)

        # 4. Multiply attention maps from first → last layer
        rollout = attn_norm[0]
        for l in range(1, num_layers):
            rollout = attn_norm[l] @ rollout

        return rollout

    def do1(i):
        inputs = prepare_inputs(processor, i)
        input_tokens = [processor.tokenizer.decode([i]) for i in inputs["input_ids"][0]]

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_text = processor.batch_decode(
            outputs.sequences[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_text = generated_text[0].strip()

        attentions_orig = get_attention_matrix(outputs)

        query_idx = -1
        attentions = attentions_orig[:, :, query_idx, :]
        attentions = attentions.cpu().float().numpy()

        attention_rollout = get_attention_rollout(attentions_orig)
        attention_rollout = attention_rollout[query_idx, :]
        attention_rollout = attention_rollout.cpu().float().numpy()

        datum = get_datum(i)
        return {
            **datum,
            "generated-text": generated_text,
            "attentions": attentions,
            "attention-rollout": attention_rollout,
            "input-tokens": input_tokens,
        }

    with h5py.File(PATH_ATTENTIONS_H5.format(dataset_name), "a") as f:
        for i in range(10):
            print(i)
            result = do1(i)
            f.create_group(str(i))
            for k, v in result.items():
                f[str(i)].create_dataset(k, data=v)


if __name__ == "__main__":
    extract_attentions()
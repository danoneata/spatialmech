import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from tqdm import tqdm
import random
import os
from pathlib import Path


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_question(objects, prepositions, is_one_object=False):
    prep_list = sorted(list(prepositions))  # Sort for consistency
    prep_options = ", ".join(prep_list[:-1]) + f" or {prep_list[-1]}"

    if is_one_object:
        object1 = objects[0] if isinstance(objects, list) else objects
        question = f"Where is the {object1} localized in the image?. Answer with {prep_options}."
    else:
        object1, object2 = objects[0], objects[1]
        question = f"Where is the {object1} in relation to the {object2}? Answer with {prep_options}."

    return question


def identify_token_ranges(input_ids, tokenizer):
    """
    Identify vision and text token ranges in the input sequence.

    Returns:
        vision_range: (start, end) indices for vision tokens
        text_range: (start, end) indices for text tokens
    """
    # Vision tokens are typically marked by special tokens
    # For Qwen3-VL, vision tokens are usually at the beginning after BOS

    # Find special tokens
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Convert to list for easier handling
    ids_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids

    # Find vision range
    vision_start = None
    vision_end = None

    for i, token_id in enumerate(ids_list):
        if token_id == vision_start_id:
            vision_start = i
        elif token_id == vision_end_id and vision_start is not None:
            vision_end = i + 1  # Include the end token
            break

    # If no explicit markers, assume vision tokens are in a contiguous block near the start
    if vision_start is None:
        # Look for the first im_start token - text usually starts after this
        for i, token_id in enumerate(ids_list):
            if token_id == im_start_id:
                vision_start = 0
                vision_end = i
                break

    # Text range starts after vision tokens
    if vision_end is not None:
        text_start = vision_end
        text_end = len(ids_list)
    else:
        # Fallback: assume first 1200 tokens are vision (typical for Qwen3-VL)
        vision_start = 0
        vision_end = min(1200, len(ids_list) - 20)  # Leave space for text
        text_start = vision_end
        text_end = len(ids_list)

    return (vision_start, vision_end), (text_start, text_end)


class AttentionKnockoutHook:
    """
    Hook to modify attention weights during generation.
    Prevents the last generated token from attending to specified regions (e.g., image tokens).
    """

    def __init__(self, vision_range, knockout_mode="block_image"):
        """
        Args:
            vision_range: (start, end) tuple for vision token indices
            knockout_mode: Type of knockout to apply
                - "block_image": Block last token from attending to image
        """
        self.vision_range = vision_range
        self.knockout_mode = knockout_mode
        self.hooks = []

    def create_attention_hook(self, layer_idx):
        """Create a forward hook for a specific layer."""

        def hook_fn(module, args, kwargs, output):
            # output is typically (hidden_states, attention_weights, past_key_value)
            # We need to modify attention_weights if they exist

            # Get attention weights from output
            if isinstance(output, tuple) and len(output) > 1:
                hidden_states = output[0]
                attn_weights = output[1]

                if attn_weights is not None and self.knockout_mode == "block_image":
                    # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                    batch_size, num_heads, seq_len, key_len = attn_weights.shape

                    # Only modify the last token's attention (last row)
                    # Block attention to vision tokens
                    vision_start, vision_end = self.vision_range

                    # Set attention to vision tokens to very small value (effectively zero)
                    attn_weights[:, :, -1, vision_start:vision_end] = -1e10

                    # Renormalize
                    attn_weights[:, :, -1, :] = torch.softmax(attn_weights[:, :, -1, :], dim=-1)

            return output

        return hook_fn

    def register_hooks(self, model):
        """Register hooks on all attention layers."""
        if len(self.hooks) > 0:
            # Already registered
            return

        print(f"Registering knockout hooks with mode: {self.knockout_mode}")

        # For Qwen3VL, layers are in model.model.language_model.layers
        layers = model.model.language_model.layers

        # Register hooks on attention layers
        for layer_idx, layer in enumerate(layers):
            # Hook into the self-attention module
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(
                    self.create_attention_hook(layer_idx),
                    with_kwargs=True
                )
                self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_knockout_inference(
    dataset_path,
    dataset_name,
    model_path,
    knockout_mode="block_image",
    device="cuda",
    output_dir="./output"
):
    """
    Run inference with attention knockout.

    Args:
        knockout_mode: Type of knockout to apply
            - "block_image": Block last token from attending to image (main experiment)
            - "normal": Normal inference without knockout (baseline)
    """
    # Set seeds for deterministic behavior
    set_seeds(42)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Preprocess: Remove '_of' from preposition labels
    preposition_mapping = {
        "right_of": "right",
        "left_of": "left",
        "in-front_of": "front",
        "behind": "behind",
    }

    def map_preposition(example):
        example["preposition"] = preposition_mapping.get(
            example["preposition"], example["preposition"]
        )
        return example

    print("Preprocessing dataset: mapping prepositions...")
    dataset = dataset.map(map_preposition)
    print(f"Prepositions mapped: {preposition_mapping}")

    # Check if this is a one-object dataset
    is_one_object = "one" in dataset_name.lower()

    # Get unique prepositions
    prepositions = set(dataset["preposition"])
    print(f"Prepositions in dataset: {prepositions}")

    # Load model and processor
    print(f"Loading Qwen3-VL from {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"  # Required for attention modification
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # Prepare results storage
    results = []

    # Process each sample
    print(f"Running inference with knockout_mode={knockout_mode}...")
    for idx, sample in enumerate(tqdm(dataset)):
        # Extract sample data
        image = sample["image"]
        caption_correct = sample["caption_correct"]
        caption_incorrect = sample["caption_incorrect"]
        preposition = sample["preposition"]
        objects = sample["objects"]

        # Create question
        question = create_question(objects, prepositions, is_one_object)

        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Identify token ranges for knockout
        input_ids = inputs["input_ids"][0]
        vision_range, text_range = identify_token_ranges(input_ids, tokenizer)

        # Run inference with or without knockout
        try:
            # Register knockout hooks if needed
            knockout_hook = None
            if knockout_mode != "normal":
                knockout_hook = AttentionKnockoutHook(vision_range, knockout_mode)
                knockout_hook.register_hooks(model)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    output_attentions=True,  # Required for hooks to work
                )

            # Remove hooks after generation
            if knockout_hook is not None:
                knockout_hook.remove_hooks()

            # Decode only the generated part (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_text = processor.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            generated_text = "ERROR"
            if knockout_hook is not None:
                knockout_hook.remove_hooks()

        # Store results
        result = {
            "sample_idx": idx,
            "knockout_mode": knockout_mode,
            "question": question,
            "generated_answer": generated_text,
            "caption_correct": caption_correct,
            "caption_incorrect": (
                caption_incorrect
                if isinstance(caption_incorrect, str)
                else ", ".join(caption_incorrect)
            ),
            "preposition": preposition,
            "objects": objects if is_one_object else f"{objects[0]}, {objects[1]}",
            "object1": (
                objects[0]
                if not is_one_object
                else objects[0] if isinstance(objects, list) else objects
            ),
            "object2": (
                objects[1]
                if not is_one_object and isinstance(objects, list) and len(objects) > 1
                else None
            ),
        }
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate accuracy (check if preposition is in generated answer)
    if not is_one_object:
        df["correct"] = df.apply(
            lambda row: row["preposition"].lower() in row["generated_answer"].lower(),
            axis=1,
        )
        accuracy = df["correct"].mean()
        print(f"\nAccuracy with {knockout_mode}: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{dataset_name}_knockout_{knockout_mode}_results.csv")
    output_parquet = os.path.join(output_dir, f"{dataset_name}_knockout_{knockout_mode}_results.parquet")

    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"Results saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_parquet}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with attention knockout experiments"
    )
    parser.add_argument(
        "--dataset",
        default="controlled_b",
        type=str,
        choices=[
            "controlled_a",
            "controlled_b",
            "coco_one",
            "coco_two",
        ],
        help="Dataset to use for inference"
    )
    parser.add_argument(
        "--knockout_mode",
        default="block_image",
        type=str,
        choices=["normal", "block_image"],
        help="Knockout mode: normal (baseline), block_image (prevent last token from seeing image)"
    )
    args = parser.parse_args()

    # Configuration
    base_path = "/leonardo_work/EUHPC_D27_102/spatialmech/dataset"
    model_path = "/leonardo_work/EUHPC_D27_102/compmech/models/Qwen3-VL-4B-Instruct"
    output_dir = "./output"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset configurations
    datasets = {
        "controlled_a": "controlled_a.hf",
        "controlled_b": "controlled_b.hf",
        "coco_one": "coco_qa_one_obj.hf",
        "coco_two": "coco_qa_two_obj.hf",
    }
    dataset_name = datasets[args.dataset]

    # Run inference
    dataset_path = os.path.join(base_path, dataset_name)
    print(f"Processing {dataset_name} with knockout_mode={args.knockout_mode}")
    print(f"\n{'='*80}")

    try:
        df = run_knockout_inference(
            dataset_path,
            dataset_name,
            model_path,
            knockout_mode=args.knockout_mode,
            device=device,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print summary
    print("\nSummary")
    print(f"{'='*80}")
    if "correct" in df.columns:
        accuracy = df["correct"].mean()
        print(f"{dataset_name} ({args.knockout_mode}): {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")
    else:
        print(f"{dataset_name}: {len(df)} samples processed")


if __name__ == "__main__":
    main()

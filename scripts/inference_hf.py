import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import random
import os


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


def run_inference(
    dataset_path, dataset_name, model_path, device="cuda", output_dir="./output"
):
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
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    model.eval()

    # Prepare results storage
    results = []

    # Process each sample
    print("Running inference...")
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

        # Run inference
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,  # Greedy decoding for deterministic results
                )

                # Decode only the generated part (skip input tokens)
                input_length = inputs["input_ids"].shape[1]
                generated_text = processor.batch_decode(
                    generated_ids[:, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            generated_text = "ERROR"

        # Store results
        result = {
            "sample_idx": idx,
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
        print(f"\nAccuracy: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{dataset_name}_results.csv")
    output_parquet = os.path.join(output_dir, f"{dataset_name}_results.parquet")

    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"Results saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_parquet}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="controlled_a",
        type=str,
        choices=[
            "controlled_a",
            "controlled_b",
            "coco_one",
            "coco_two",
        ],
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
    # Run inference on each dataset
    all_results = {}
    dataset_path = os.path.join(base_path, dataset_name)
    print(f"Processing {dataset_name}")
    print(f"\n{'='*80}")

    try:
        df = run_inference(dataset_path, dataset_name, model_path, device, output_dir)
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback

        traceback.print_exc()

    # Print summary
    print("Summary")
    print(f"{'='*80}")
    if "correct" in df.columns:
        accuracy = df["correct"].mean()
        print(f"{dataset_name}: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")
    else:
        print(f"{dataset_name}: {len(df)} samples processed")


if __name__ == "__main__":
    main()

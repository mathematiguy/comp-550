import click
import torch
import pandas as pd
import re
import json
from transformers import AutoTokenizer, pipeline
from tqdm.auto import tqdm
from nltk.corpus import wordnet as wn
import logging
from loader import load_key
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

# Define functions with docstrings and logging


def synset_from_key(sense_key):
    """
    Retrieve the synset name from a sense key.
    """
    lemma = wn.lemma_from_key(sense_key)
    return lemma.synset().name()


def get_synset_definition(synset_id):
    """
    Retrieve the definition of a synset.
    """
    synset = wn.synset(synset_id)
    return synset.definition()


def get_synset_examples(synset_id):
    """
    Retrieve examples for a given synset.
    """
    synset = wn.synset(synset_id)
    return synset.examples()


def generate_prompt(word, definition, examples):
    """
    Generate a prompt for text generation.
    """
    prompt = (
        f'Below is one definition for "{word}" with examples.\n'
        f"Definition ({word}): {definition}\n"
    )
    if examples:
        prompt += "Examples:\n- " + "\n- ".join(examples) + "\n"
    prompt += "Please generate 10 more example sentences.\n"
    return prompt


def extract_examples(text):
    """
    Extract examples from generated text.
    """
    matches = re.findall(r"(\*|•|-|\d[\).])\s*(.*)", text)
    return [m[1] for m in matches]


def build_dataframe(dev_key):
    """
    Build the initial dataframe from the WordNet data.
    """
    logging.info("Building the DataFrame from WordNet data")
    seeds = sorted(set(key for keys in dev_key.values() for key in keys))
    seed_data = pd.DataFrame({"sense_key": seeds})

    seed_data["synset_id"] = seed_data["sense_key"].apply(synset_from_key)
    seed_data["word"] = seed_data["synset_id"].str.split(".").apply(lambda x: x[0])
    seed_data["definition"] = seed_data["synset_id"].apply(get_synset_definition)
    seed_data["examples"] = seed_data["synset_id"].apply(get_synset_examples)
    seed_data["prompt"] = seed_data.apply(
        lambda x: generate_prompt(x["word"], x["definition"], x["examples"]), axis=1
    )

    return seed_data


def generate_text(seed_data, pipeline, tokenizer, batch_size):
    """
    Generate text using the provided pipeline and batch size.
    """
    logging.info("Generating text using the pipeline")
    prompts = seed_data["prompt"].tolist()
    dataset = Dataset.from_dict({"prompt": prompts})

    results = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]["prompt"]
        batch_results = pipeline(
            batch,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=500,
        )
        results.extend(batch_results)

    seed_data["generated_text"] = [r[0]["generated_text"] for r in results]
    seed_data["generated_examples"] = seed_data["generated_text"].apply(
        extract_examples
    )


def load_seed_dataset(csv_path):
    """
    Load the DataFrame from CSV file.
    """
    logging.info(f"Saving DataFrame to {csv_path}")
    seed_data = pd.read_csv(csv_path)
    seed_data["examples"] = seed_data["examples"].apply(json.loads)
    seed_data["generated_examples"] = seed_data["generated_examples"].apply(json.loads)
    return seed_data


def save_seed_dataset(seed_data, csv_path):
    """
    Save the DataFrame to a CSV file.
    """
    logging.info(f"Saving DataFrame to {csv_path}")
    seed_data_to_save = seed_data.copy()
    seed_data_to_save["examples"] = seed_data_to_save["examples"].apply(json.dumps)
    seed_data_to_save["generated_examples"] = seed_data_to_save[
        "generated_examples"
    ].apply(json.dumps)
    seed_data_to_save.to_csv(csv_path, index=False)


@click.command()
@click.option(
    "--model_path",
    default="/network/weights/llama.var/llama2/Llama-2-7b-chat-hf",
    help="Model path for the text generation.",
)
@click.option("--batch_size", default=14, help="Batch size for text generation.")
@click.option(
    "--csv_path",
    default="data/seed_set_data.csv",
    help="CSV file path to save the data.",
)
def cli(model_path, batch_size, csv_path):
    """
    CLI to generate and save data using a text generation model.
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load keys
    dev_key, test_key = load_key("data/wordnet.en.key")

    # Build the DataFrame
    seed_data = build_dataframe(dev_key)

    # Run text generation
    generate_text(seed_data, text_generation_pipeline, tokenizer, batch_size)

    # Save the DataFrame
    save_seed_dataset(seed_data, csv_path)


if __name__ == "__main__":
    cli()

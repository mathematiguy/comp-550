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
    prompt += "Please generate 5 more example sentences.\n"
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
    lemma_sense_keys = sorted(set(key for keys in dev_key.values() for key in keys))
    lemma_sense_key_data = pd.DataFrame({"sense_key": lemma_sense_keys})

    lemma_sense_key_data["synset_id"] = lemma_sense_key_data["sense_key"].apply(
        synset_from_key
    )
    lemma_sense_key_data["word"] = (
        lemma_sense_key_data["synset_id"].str.split(".").apply(lambda x: x[0])
    )
    lemma_sense_key_data["definition"] = lemma_sense_key_data["synset_id"].apply(
        get_synset_definition
    )
    lemma_sense_key_data["examples"] = lemma_sense_key_data["synset_id"].apply(
        get_synset_examples
    )
    lemma_sense_key_data["prompt"] = lemma_sense_key_data.apply(
        lambda x: generate_prompt(x["word"], x["definition"], x["examples"]), axis=1
    )

    return lemma_sense_key_data


def generate_text(lemma_sense_key_data, pipeline, tokenizer, batch_size):
    """
    Generate text using the provided pipeline and batch size.
    """
    logging.info("Generating text using the pipeline")
    prompts = lemma_sense_key_data["prompt"].tolist()
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
            max_length=400,
        )
        results.extend(batch_results)

    lemma_sense_key_data["generated_text"] = [r[0]["generated_text"] for r in results]
    lemma_sense_key_data["generated_examples"] = lemma_sense_key_data[
        "generated_text"
    ].apply(extract_examples)


def save_dataframe(lemma_sense_key_data, csv_path):
    """
    Save the DataFrame to a CSV file.
    """
    logging.info(f"Saving DataFrame to {csv_path}")
    lemma_sense_key_data_to_save = lemma_sense_key_data.copy()
    lemma_sense_key_data_to_save["examples"] = lemma_sense_key_data_to_save[
        "examples"
    ].apply(json.dumps)
    lemma_sense_key_data_to_save["generated_examples"] = lemma_sense_key_data_to_save[
        "generated_examples"
    ].apply(json.dumps)
    lemma_sense_key_data_to_save.to_csv(csv_path, index=False)


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
    lemma_sense_key_data = build_dataframe(dev_key)

    # Run text generation
    generate_text(lemma_sense_key_data, text_generation_pipeline, tokenizer, batch_size)

    # Save the DataFrame
    save_dataframe(lemma_sense_key_data, csv_path)


if __name__ == "__main__":
    cli()

import os
import random
from pathlib import Path

import click
import yaml
from llama import Llama, Dialog

# Set environment variables if not set
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', str(random.randint(2000, 65535)))  # Choose a random port
os.environ.setdefault('RANK', "0")
os.environ.setdefault('WORLD_SIZE', "1")


@click.command()
@click.option('--bird', required=True, help='Bird for fact generation.')
@click.option('--bird-name', required=True, help='Bird name for fact generation.')
@click.option('--output-path', required=True, type=click.Path(), help='Path to store the generated facts.')
@click.option('--ckpt-dir', default='models/llama-2-7b-chat', help='Path to model checkpoint directory.')
@click.option('--tokenizer-path', default='models/llama/tokenizer.model', help='Path to tokenizer model.')
@click.option('--dialog-template-path', default='data/prompts/bird-facts.yaml', help='Path to the bird dialog template.')
@click.option('--num-real-facts', default=50, help='Number of real facts to generate')
@click.option('--num-fake-facts', default=50, help='Number of fake facts to generate')
@click.option('--max-gen-len', default=None, help='Maximum generation length.')
@click.option('--temperature', default=0.6, help='Temperature parameter for generation.')
@click.option('--top-p', default=0.9, help='Top-p parameter for generation.')
@click.option('--max-seq-len', default=4096, help='Maximum sequence length for the model.')
@click.option('--max-batch-size', default=8, help='Maximum batch size for the model.')
def generate_bird_facts(bird, bird_name, output_path, ckpt_dir, tokenizer_path, dialog_template_path, num_real_facts, num_fake_facts, max_gen_len, temperature, top_p, max_seq_len, max_batch_size):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open(dialog_template_path, 'r') as f:
        bird_dialog_yaml = f.read()

    bird_dialog_yaml = bird_dialog_yaml.format(
        bird_name=bird_name,
        num_real_facts=num_real_facts,
        num_fake_facts=num_fake_facts
    )
    bird_dialog = yaml.load(bird_dialog_yaml, Loader=yaml.FullLoader)

    results = generator.chat_completion(
        bird_dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    with open(output_path, 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    generate_bird_facts()

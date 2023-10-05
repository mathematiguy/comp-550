import os
import re
import yaml
import pandas as pd
import click

def extract_facts(text):
    pattern = r'\[START\](.*?)\[END\]'
    facts = re.findall(pattern, text, re.DOTALL)
    facts = [fact.strip() for fact in facts]
    return facts

def load_and_compile_bird_data(directory):
    all_bird_data = {}
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for filename in filenames:
        if filename.startswith('.'):
            continue

        bird_name = filename.replace('_facts.yaml', '')

        with open(os.path.join(directory, filename), 'r') as file:
            bird_data = yaml.safe_load(file)

            real, fake = bird_data
            real = real['generation']['content']
            fake = fake['generation']['content']
            all_bird_data[bird_name] = {'real': extract_facts(real), 'fake': extract_facts(fake)}

    return all_bird_data

def nested_dict_to_dataframe(data):
    rows = []
    for bird, facts in data.items():
        for fact_type, fact_list in facts.items():
            for fact in fact_list:
                rows.append({
                    'Bird': bird,
                    'Fact Type': fact_type,
                    'Fact': fact
                })

    df = pd.DataFrame(rows)
    return df

@click.command()
@click.option('--directory', default='data/generations', help='Directory containing bird fact YAML files.')
@click.option('--output', default='data/bird_data.csv', help='Directory containing bird fact YAML files.')
def main(directory, output):
    bird_facts = load_and_compile_bird_data(directory)
    bird_data = nested_dict_to_dataframe(bird_facts)
    bird_data.to_csv(output, index=False)
    print(f"Data saved to {output}")

if __name__ == '__main__':
    main()

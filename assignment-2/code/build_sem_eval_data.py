import click
import pandas as pd
from wsd import preprocess, lesk


def build_sem_eval_data():
    """
    Constructs a dataframe out of the SemEval 2013 dataset.
    """

    sem_eval_data = []
    for key, instance in dev_instances.items():
        id = instance.id
        lemma = instance.lemma
        context = instance.context

        processed_context = preprocess(context)

        # Use lesk's algorithm to guess the synset
        synset = lesk(lemma, context)

        # Get the sense-keys for the predicted synset
        preds = set(lemma.key() for lemma in synset.lemmas())

        # Extract the synset number from the sense-key
        targets = lsk_to_sn[id]

        # Calculate if there is any overlap between the predicted sense and the target
        match = len(preds & targets) > 0

        sem_eval_data.append(
            dict(
                id=id,
                lemma=lemma,
                context=context,
                processed_context=processed_context,
                synset=synset,
                preds=preds,
                targets=targets,
                match=match,
            )
        )

    # Save the DataFrame to a file
    df.to_csv(output_file)


@click.command()
@click.option("--input_file", default="input.csv", help="Input file path.")
@click.option("--output_file", default="output.csv", help="Output file path.")
def cli(input_file, output_file):
    build_sem_eval_data(input_file, output_file)
    click.echo(f"Data saved to {output_file}")


if __name__ == "__main__":
    cli()

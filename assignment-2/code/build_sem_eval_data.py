import click
import pandas as pd
from loader import load_instances, load_key
from wsd import preprocess, wordnet_lesk


def get_instances():
    data_f = "data/multilingual-all-words.en.xml"
    key_f = "data/wordnet.en.key"
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    return dev_instances, test_instances


def get_lemma_sense_key_to_synset_number_correspondence():
    wordnet_key_file = "data/wordnet.en.key"
    lsk_to_sn = {}
    with open(wordnet_key_file, "r") as f:
        for line in f.read().strip().split("\n"):
            line = line.strip()
            _, lsk, sn = line.split(" ", 2)
            lsk_to_sn[lsk] = set(sn.split(" "))
    return lsk_to_sn


def build_sem_eval_data(csv_path):
    """
    Constructs a dataframe out of the SemEval 2013 dataset.
    """

    dev_instances, test_instances = get_instances()

    lsk_to_sn = get_lemma_sense_key_to_synset_number_correspondence()

    sem_eval_data = []
    for test_set, instances in zip(["dev", "test"], [dev_instances, test_instances]):
        for key, instance in instances.items():
            id = instance.id
            lemma = instance.lemma
            context = instance.context

            processed_context = preprocess(context)

            # Use lesk's algorithm to guess the synset
            synset = wordnet_lesk(lemma, context)

            # Get the sense-keys for the predicted synset
            preds = set(lemma.key() for lemma in synset.lemmas())

            # Extract the synset number from the sense-key
            targets = lsk_to_sn[id]

            # Calculate if there is any overlap between the predicted sense and the target
            match = len(preds & targets) > 0

            sem_eval_data.append(
                dict(
                    id=id,
                    test_set=test_set,
                    lemma=lemma,
                    context=context,
                    processed_context=processed_context,
                    synset=synset,
                    preds=preds,
                    targets=targets,
                    match=match,
                )
            )

    return pd.DataFrame(sem_eval_data)


def calculate_lesk_algorithm_accuracy(sem_eval_data):
    test_acc = 100 * sem_eval_data.loc[sem_eval_data.test_set == "test", "match"].mean()
    dev_acc = 100 * sem_eval_data.loc[sem_eval_data.test_set == "dev", "match"].mean()
    return test_acc, dev_acc


@click.command()
@click.option(
    "--csv_path",
    default="data/sem_eval_data.csv",
    help="Path to save sem_eval_data.csv",
)
def cli(csv_path):
    sem_eval_data = build_sem_eval_data(csv_path)

    test_acc, dev_acc = calculate_lesk_algorithm_accuracy(sem_eval_data)
    click.echo(f"Lesk Algorithm's Test Accuracy: {test_acc:.1f}%")
    click.echo(f"Lesk Algorithm's Dev Accuracy: {dev_acc:.1f}%")

    # Save the DataFrame to a file
    sem_eval_data.to_csv(csv_path, index=False)

    click.echo(f"Data saved to {csv_path}")


if __name__ == "__main__":
    cli()

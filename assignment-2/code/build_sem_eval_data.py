import click
import pandas as pd
import logging

import nltk
from loader import load_instances, load_key
from wsd import preprocess, wordnet_lesk, most_frequent_synset

# Setup logging
logging.basicConfig(level=logging.INFO)


def get_instances():
    """
    Loads the SemEval dataset instances and keys from files.
    """
    logging.info("Loading instances and keys")
    data_f = "data/multilingual-all-words.en.xml"
    key_f = "data/wordnet.en.key"
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # Filter instances by keys
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    return dev_instances, test_instances


def get_lemma_sense_key_to_synset_number_correspondence():
    """
    Creates a mapping from lemma sense keys to synset numbers.
    """
    logging.info("Mapping lemma sense keys to synset numbers")
    wordnet_key_file = "data/wordnet.en.key"
    lsk_to_sn = {}
    with open(wordnet_key_file, "r") as f:
        for line in f.read().strip().split("\n"):
            _, lsk, sn = line.strip().split(" ", 2)
            lsk_to_sn[lsk] = set(sn.split(" "))
    return lsk_to_sn


def build_sem_eval_data(csv_path):
    """
    Constructs a dataframe from the SemEval 2013 dataset.
    """
    logging.info("Building SemEval dataset")
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

    df = pd.DataFrame(sem_eval_data)
    df["most_frequent_synset"] = df.lemma.apply(most_frequent_synset)
    df["nltk_pred_synset"] = df.apply(
        lambda x: set(
            lemma.key() for lemma in nltk.wsd.lesk(x.context, x.lemma).lemmas()
        ),
        axis=1,
    )

    df.to_csv(csv_path, index=False)
    logging.info(f"Dataframe saved to {csv_path}")
    return df


def calculate_lesk_algorithm_accuracy(sem_eval_data):
    """
    Calculates and returns the Lesk Algorithm's accuracy on the test and dev sets.
    """
    test_acc = 100 * sem_eval_data.loc[sem_eval_data.test_set == "test", "match"].mean()
    dev_acc = 100 * sem_eval_data.loc[sem_eval_data.test_set == "dev", "match"].mean()
    return dev_acc, test_acc


def calculate_most_frequent_synset_accuracy(sem_eval_data):
    dev_acc = (
        sem_eval_data.loc[sem_eval_data.test_set == "dev", :]
        .apply(lambda x: len(x.most_frequent_synset & x.targets) > 0, axis=1)
        .mean()
    )
    test_acc = (
        sem_eval_data.loc[sem_eval_data.test_set == "test", :]
        .apply(lambda x: len(x.most_frequent_synset & x.targets) > 0, axis=1)
        .mean()
    )
    return dev_acc, test_acc


def calculate_nltk_lesk_algorithm_accuracy(sem_eval_data):
    dev_acc = (
        sem_eval_data.loc[sem_eval_data.test_set == "dev", :]
        .apply(lambda x: len(x.nltk_pred_synset & x.targets) > 0, axis=1)
        .mean()
    )
    test_acc = (
        sem_eval_data.loc[sem_eval_data.test_set == "test", :]
        .apply(lambda x: len(x.nltk_pred_synset & x.targets) > 0, axis=1)
        .mean()
    )
    return dev_acc, test_acc


@click.command()
@click.option(
    "--csv_path",
    default="data/sem_eval_data.csv",
    help="Path to save sem_eval_data.csv",
)
def cli(csv_path):
    """
    CLI command to process SemEval data and calculate accuracy.
    """
    sem_eval_data = build_sem_eval_data(csv_path)
    dev_acc, test_acc = calculate_lesk_algorithm_accuracy(sem_eval_data)
    mfs_dev_acc, mfs_test_acc = calculate_most_frequent_synset_accuracy(sem_eval_data)
    nltk_dev_acc, nltk_test_acc = calculate_nltk_lesk_algorithm_accuracy(sem_eval_data)

    click.echo(f"Lesk Algorithm's Test Accuracy:\t\t\t{test_acc:.1f}%")
    click.echo(f"Lesk Algorithm's Dev Accuracy:\t\t\t{dev_acc:.1f}%")

    click.echo(f"Most Frequent Synset Dev Accuracy:\t{100*mfs_dev_acc:.1f}%")
    click.echo(f"Most Frequent Synset Test Accuracy:\t{100*mfs_test_acc:.1f}%")

    click.echo(f"NLTK Lesk Dev Accuracy:\t\t\t\t\t\t\t{100*nltk_dev_acc:.1f}%")
    click.echo(f"NLTK Lesk Test Accuracy:\t\t\t\t\t\t{100*nltk_test_acc:.1f}%")


if __name__ == "__main__":
    cli()

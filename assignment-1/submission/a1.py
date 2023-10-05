import os
import re
import yaml
import click

import pandas as pd
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def extract_facts(text):
    pattern = r'\[START\](.*?)\[END\]'
    facts = re.findall(pattern, text, re.DOTALL)
    facts = [fact.strip() for fact in facts]
    return facts


def load_data_from_files():
    rows = []

    # Load facts from facts.txt and tag them as "real"
    with open('data/facts.txt', 'r') as f:
        for line in f:
            rows.append({
                'Fact Type': 'real',
                'Fact': line.strip()
            })

    # Load facts from fakes.txt and tag them as "fake"
    with open('data/fakes.txt', 'r') as f:
        for line in f:
            rows.append({
                'Fact Type': 'fake',
                'Fact': line.strip()
            })

    df = pd.DataFrame(rows)
    return df

def perform_train_test_split(bird_data):
    # Create a new column for stratification
    bird_data['strata'] = bird_data['Fact Type']

    # Splitting the dataset 60/40 first
    train, temp = train_test_split(bird_data, train_size=0.6, stratify=bird_data['strata'])

    # Splitting the 40% dataset into two equal parts for dev and test
    dev, test = train_test_split(temp, train_size=0.5, stratify=temp['strata'])

    # Now, you can drop the strata as it's no longer needed
    bird_data.drop(columns=['strata'], inplace=True)
    train.drop(columns=['strata'], inplace=True)
    dev.drop(columns=['strata'], inplace=True)
    test.drop(columns=['strata'], inplace=True)

    # Reset the indices
    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Training set size: {len(train)}")
    print(f"Development set size: {len(dev)}")
    print(f"Test set size: {len(test)}")

    return train, dev, test


class ModelFactory:
    def __init__(self):
        self.vocab = None  # This will store our vocabulary
        self.classifier = None

    def preprocess_text(self, text, methods):
        tokens = word_tokenize(text)

        if methods.get("stopword_removal"):
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]

        if methods.get("lemmatization"):
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if methods.get("stemming"):
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]

        if methods.get("n-grams"):
            ngram_n = methods.get("n-grams")
            n_grams_tokens = list(ngrams(tokens, ngram_n))
            tokens.extend(['_'.join(gram) for gram in n_grams_tokens])

        if methods.get("treebank_pos"):
            tokens = [f"{word}_{tag}" for word, tag in nltk.pos_tag(tokens)]

        return tokens

    def extract_features(self, dataset, methods, build_vocab=True):
        all_tokens = []
        for text in dataset:
            tokens = self.preprocess_text(text, methods)
            all_tokens.extend(tokens)

        if build_vocab:
            vocab_limit = methods.get("vocab_limit", 1000)  # Default to 1000 if not provided
            self.vocab = [item[0] for item in Counter(all_tokens).most_common(vocab_limit)]

        featuresets = np.zeros((len(dataset), len(self.vocab)))
        for idx, text in enumerate(dataset):
            tokens = self.preprocess_text(text, methods)
            for token in tokens:
                if token in self.vocab:
                    featuresets[idx, self.vocab.index(token)] = 1

        return featuresets

    def train(self, train_texts, train_labels, preprocess_methods):
        X_train = self.extract_features(train_texts, preprocess_methods, build_vocab=True)

        classifier_type = preprocess_methods.get("classifier_type", "sgd")

        if classifier_type == "sgd":
            self.classifier = SGDClassifier()
        elif classifier_type == "logistic":
            self.classifier = LogisticRegression()
        elif classifier_type == "svm":
            self.classifier = SVC(kernel='linear')
        elif classifier_type == "naive_bayes":
            self.classifier = MultinomialNB()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classifier.fit(X_train, train_labels)


    def predict(self, texts, methods):
        X = self.extract_features(texts, methods, build_vocab=False)
        return self.classifier.predict(X)


def sample_hyperparameters(seed=None):

    if seed is not None:
        np.random.seed(seed)

    # For Bernoulli, we use numpy's randint which returns 0 or 1
    lemmatization = np.random.randint(2)
    stopword_removal = np.random.randint(2)
    stemming = np.random.randint(2)
    treebank_pos = np.random.randint(2)

    # For n-grams, using randint between 1 and 6 will give a value in [1, 2, 3, 4, 5]
    n_grams = np.random.randint(1, 6)

    # For vocab_limit, sample from a lognormal distribution. The parameters for this lognormal distribution
    # are set so that it peaks around 500 and the large majority of its mass is before 1300
    mu, sigma = np.log(500), 0.5
    vocab_limit = int(np.random.lognormal(mu, sigma))

    # Resample if the value exceeds 1300
    while vocab_limit > 1300:
        vocab_limit = int(np.random.lognormal(mu, sigma))

    # For classifier choice, generate a random integer and map it to a classifier
    classifier_mapping = {
        0: "sgd",
        1: "logistic",
        2: "svm",
        3: "naive_bayes"
    }
    classifier_type = classifier_mapping[np.random.randint(4)]

    return {
        "lemmatization": bool(lemmatization),
        "stopword_removal": bool(stopword_removal),
        "n-grams": n_grams,
        "stemming": bool(stemming),
        "treebank_pos": bool(treebank_pos),
        "vocab_limit": vocab_limit,
        "classifier_type": classifier_type
    }


def run_trial(trial_num):
    model_factory = ModelFactory()
    methods = sample_hyperparameters(seed=trial_num)
    model_factory.train(train_texts=train['Fact'], train_labels=train['Fact Type'], preprocess_methods=methods)
    predictions = model_factory.predict(dev['Fact'], methods)
    accuracy = np.sum(predictions == dev['Fact Type']) / len(dev['Fact Type'])
    return {**methods, 'accuracy': accuracy}


@click.command()
@click.option('--directory', default='data/generations', help='Directory containing bird fact YAML files.')
@click.option('--output', default='data/bird_data.csv', help='Directory containing bird fact YAML files.')
@click.option('--bird-data-path', default='data/bird_data.csv', type=str, help='Path to bird_data.csv')
@click.option('--num-trials', default=100, type=int, help='Number of trials to run.')
@click.option('--path', default='trial_results.csv', type=str, help='Path to save the trial results.')
def main(directory, output, bird_data_path, num_trials, path):

    bird_data = load_data_from_files()

    global train, dev, test
    train, dev, test = perform_train_test_split(bird_data)

    # Introduce parallelism using ProcessPoolExecutor
    if not os.path.exists('data/trials.csv'):
        with ProcessPoolExecutor() as executor:
            trials = list(tqdm(executor.map(run_trial, range(num_trials)), total=num_trials))

        trial_df = pd.DataFrame(trials)
        trial_df.to_csv(path, index=False)
        print(f"Results saved to {path}")

    trial_df = pd.read_csv('data/trials.csv')

    # Create dummy variables for classifier_type
    classifier_dummies = pd.get_dummies(trial_df['classifier_type'], prefix='classifier')

    # Concatenate the original dataframe with the dummy columns
    trial_df = pd.concat([trial_df, classifier_dummies], axis=1)

    # Drop the original classifier_type column
    trial_df = trial_df.drop(columns=['classifier_type'])

    # Creating the design matrix X and the target vector y
    X = trial_df[['lemmatization', 'stopword_removal', 'n-grams', 'stemming', 'treebank_pos', 'vocab_limit',
                  'classifier_sgd', 'classifier_logistic', 'classifier_svm', 'classifier_naive_bayes']].values
    y = trial_df['accuracy'].values

    # Adding an intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    X = X.astype(float)

    # Create an instance of the scaler
    scaler = StandardScaler()

    # Scaling the features, excluding the intercept
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    # Now, fit the regression model
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Define the feature names
    feature_names = ["Intercept", "Lemmatization", "Stopword Removal", "N-grams", "Stemming", "Treebank POS",
                     "Vocab Limit", "Classifier_SGD", "Classifier_Logistic", "Classifier_SVM", "Classifier_Naive_Bayes"]

    # Create a DataFrame
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort the DataFrame based on the magnitude of coefficients for better interpretation
    coefficients_df = coefficients_df.reindex(coefficients_df.Coefficient.abs().sort_values(ascending=False).index)

    print(coefficients_df)


if __name__ == '__main__':
    main()

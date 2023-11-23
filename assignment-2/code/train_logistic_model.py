
"""
This script trains a logistic regression model for Word Sense Disambiguation.
"""

import sys
import json
import click
import joblib
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor

# Add the code directory to the system path
sys.path.insert(0, '../code')

# Load necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Import custom modules
from build_seed_set import load_seed_dataset

def build_seed_examples(seed_dataset_path):
    # Load the seed dataset
    seed_dataset = load_seed_dataset(seed_dataset_path)
    
    # Preprocess the seed dataset
    seed_examples = seed_dataset[['word', 'synset_id', 'generated_examples', 'examples']].copy()
    seed_examples['text'] = seed_examples.apply(lambda x: x.examples + x.generated_examples, axis=1)
    seed_examples = seed_examples.loc[:, ['word', 'synset_id', 'text']]
    seed_examples = seed_examples.explode('text')
    seed_examples = seed_examples[seed_examples.apply(lambda x: x.word in x.text, axis=1)]
    seed_examples = seed_examples.reset_index(drop=True)

    return seed_examples

class ModelFactory:
    """
    Factory class for creating and evaluating a logistic regression model.
    """
    def __init__(self, texts, labels, tokenizer, stop_words, C=1e12, class_weight='balanced', test_size=0.2):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.C = C
        self.class_weight = class_weight
        self.test_size = test_size

    def lemmatize_tokenize(self, text):
        """
        Tokenize and lemmatize the input text.
        """
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [self.tokenizer.lemmatize(token) for token in tokens if not token in self.stop_words]
        return lemmatized_tokens

    def train(self):
        """
        Train the logistic regression model.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=self.test_size)
        vectorizer = CountVectorizer(tokenizer=self.lemmatize_tokenize)
        X_train_counts = vectorizer.fit_transform(X_train)
        clf = MultiOutputRegressor(LogisticRegression(C=self.C, class_weight=self.class_weight))
        clf.fit(X_train_counts, y_train)
        return clf, vectorizer, X_train_counts, X_test, y_train, y_test

    def evaluate(self, clf, vectorizer, X_train_counts, X_test, y_train, y_test):
        """
        Evaluate the logistic regression model.
        """
        train_predictions = clf.predict(X_train_counts)
        preds = np.argmax(train_predictions, axis=1)
        targets = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(preds == targets)
        X_test_counts = vectorizer.transform(X_test)
        test_predictions = clf.predict(X_test_counts)
        test_preds = np.argmax(test_predictions, axis=1)
        test_targets = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_preds == test_targets)
        return train_accuracy, test_accuracy

    def save_model(self, clf, filename):
        """
        Save the trained model to disk.
        """
        joblib.dump(clf, filename)
        logging.info(f"Model saved to {filename}")

@click.command()
@click.option('--data_path', default='data/seed_set_data.csv', help='Path to the seed dataset.')
@click.option('--model_path', default='data/logistic.joblib', help='Path to the save the fitted model.')
def main(data_path, model_path):
    """
    Main function to run the script.
    """
    # Load and preprocess the dataset
    seed_examples = build_seed_examples(data_path)

    # Initialize and use the model factory
    lemmatizer = WordNetLemmatizer()
    factory = ModelFactory(
        texts=seed_examples.text,
        labels=pd.get_dummies(seed_examples['synset_id']) * 1,
        tokenizer=lemmatizer,
        stop_words=stop_words
    )
    clf, vectorizer, X_train_counts, X_test, y_train, y_test = factory.train()
    
    # Save the model
    model_filename = 'trained_model.joblib'
    factory.save_model(clf, model_path)

    # Measure model accuracy
    train_accuracy, test_accuracy = factory.evaluate(clf, vectorizer, X_train_counts, X_test, y_train, y_test)
    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()

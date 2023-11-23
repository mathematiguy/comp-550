from nltk import wsd
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import Lemma


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(sentence, stop_words=stop_words, lemmatizer=lemmatizer):
    """
    Preprocesses the sentence by lemmatizing and removing stop words and words without any alphanumeric characters.
    Assumes sentence is a list of tokenized words.
    """
    # Helper function to check for at least one alphanumeric character in a word
    contains_alnum = lambda word: any(char.isalnum() for char in word)

    # Lemmatize words, filter out stop words and words without any alphanumeric characters
    processed = {
        lemmatizer.lemmatize(w)
        for w in sentence
        if w not in stop_words and contains_alnum(w)
    }

    return processed


def wordnet_lesk(lemma, context):
    """
    Lesk's algorithm implementation.
    Assumes preprocessed_context is a set of lemmatized words without stop words.
    """
    assert isinstance(lemma, str), "Lemma is not a string"
    assert len(context) > 0, "Empty context"

    max_overlap = 0
    best_sense = None

    context = preprocess(context)

    # Obtain the synsets for the lemma
    synsets = wn.synsets(lemma)

    # Default to the most common sense if synsets are available
    if synsets:
        best_sense = synsets[0]

    for sense in synsets:
        # Preprocess the signature (definition and examples)
        signature = preprocess(
            word_tokenize(sense.definition()), stop_words, lemmatizer
        )
        for example in sense.examples():
            signature |= preprocess(word_tokenize(example), stop_words, lemmatizer)

        # The overlap is the size of the intersection
        overlap = len(context & signature)

        # Keep track of the best overlap so far
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


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

    return pd.DataFrame(sem_eval_data)

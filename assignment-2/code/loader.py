"""
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.

"""

import xml.etree.ElementTree as ET


class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        # id of the WSD instance
        self.id = my_id
        # lemma of the word whose sense is to be resolved
        self.lemma = lemma
        # lemma of all the words in the sentential context
        self.context = context
        # index of lemma within the context
        self.index = index

    def __str__(self):
        # For printing purposes.
        return f"{self.id}\t{self.lemma}\t{' '.join(self.context)}\t{self.index}"


def load_instances(f):
    """
    Load two lists of cases to perform WSD on. The structure that is returned is a dict,
    where the keys are the ids, and the values are instances of WSDInstance.
    """
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        instances = (
            dev_instances if text.attrib["id"].startswith("d001") else test_instances
        )
        for sentence in text:
            # construct sentence context
            context = [el.attrib["lemma"] for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == "instance":
                    instances[el.attrib["id"]] = WSDInstance(
                        el.attrib["id"], el.attrib["lemma"], context, i
                    )
    return dev_instances, test_instances


def load_key(f):
    """
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    """
    dev_key = {}
    test_key = {}
    with open(f) as file:
        for line in file:
            if len(line) <= 1:
                continue
            doc, my_id, sense_key = line.strip().split(" ", 2)
            if doc == "d001":
                dev_key[my_id] = sense_key.split()
            else:
                test_key[my_id] = sense_key.split()
    return dev_key, test_key


if __name__ == "__main__":
    data_f = "code/multilingual-all-words.en.xml"
    key_f = "code/wordnet.en.key"
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    # ready to use here
    print(len(dev_instances))  # number of dev instances
    print(len(test_instances))  # number of test instances

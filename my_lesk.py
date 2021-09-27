import numpy as np
import string
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.utils import resample
import statistics


def get_synset_definition(lemma):
    synsets = wordnet.synsets(lemma)
    return [synset.definition() for synset in synsets]


def get_sense_key(synset):
    "returns the sense key as a string for the given synset"
    sense_keys = [sense.key() for sense in synset.lemmas()]
    #print("synset: {}, synset.lemmas: {}".format(synset, synset.lemmas()))
    #print("sensekey: ", sense_keys)
    return sense_keys


def evaluate_accuracy(predictions, targets):
    correct = 0
    total = len(targets)

    for prediction, target in zip(predictions, targets):
        # prediction and target are lists, could have more than one answer, so check is 2 list intersect
        if set(prediction) & set(target):
            correct += 1
    accuracy = (correct / total)
    return accuracy


def _preprocess(sentence, remove_stowords=True, lemmatization=True, stemming=True):
    for punc in string.punctuation:
        sentence = sentence.replace(punc, ' ')
    sentence = sentence.lower()
    sentence = sentence.split()

    if remove_stowords:
        sentence = [w for w in sentence if w not in set(
            stopwords.words('english'))]
    if stemming:
        sentence = [PorterStemmer().stem(w) for w in sentence]
    if lemmatization:
        sentence = [WordNetLemmatizer().lemmatize(w) for w in sentence]
    #sentence = " ".join(sentence)
    # print(sentence)
    return sentence


def _inverse_frequency(definitions):
    vocabulary = [w for definition in definitions for w in definition]
    df = {w: 0. for w in vocabulary}
    N = float(len(definitions))
    for word in vocabulary:
        for definition in definitions:
            if word in definition:
                df[word] += 1.
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log(0.5 + 0.5*N / freq)
    return idf


def _get_overlap(context, definition, idf):
    overlap = 0.
    cur_context = context.copy()
    for _ in cur_context:
        w = cur_context.pop()
        try:
            w_idx = definition.index(w)
            definition.pop(w_idx)
            overlap += idf[w]
        except ValueError:
            pass
    return overlap


def my_lesk(lemma, context):
    synsets = wordnet.synsets(lemma)

    syn_definitions = get_synset_definition(lemma)

    syn_definitions = [_preprocess(definition)
                       for definition in syn_definitions]
    context = _preprocess(context)

    idf = _inverse_frequency(syn_definitions)

    best_overlap = 0
    best_synset_idx = 0

    for syn_idx, definition in enumerate(syn_definitions):
        if syn_idx >= 3:
            break
        else:
            cur_overlap = _get_overlap(context, definition, idf)
            if cur_overlap > best_overlap:
                best_synset_idx = syn_idx

    return get_sense_key(synsets[best_synset_idx])


def mylesk_run(dev_instances, dev_key, bootstrap=5):
    accuracy = []
    split_idx = len(dev_instances)//2
    for i in range(bootstrap):
        boot_instance = dict(list(dev_instances.items())[:split_idx])
        boot_key = dict(list(dev_key.items())[:split_idx])
        predictions = []
        targets = []
        for id, wsd in boot_instance.items():
            lemma = wsd.lemma.decode("utf-8")
            context = ' '.join([el.decode("utf-8") for el in wsd.context])
            pred = my_lesk(lemma, context)
            predictions.append(pred)
            targets.append(boot_key[id])
        split_idx +=split_idx
        accuracy.append(evaluate_accuracy(predictions, targets))

    return statistics.mean(accuracy)

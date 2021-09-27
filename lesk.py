
import numpy as np
import string
from nltk.wsd import lesk
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer


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
        #prediction and target are lists, could have more than one answer, so check is 2 list intersect
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
        sentence = [w for w in sentence if w not in set(stopwords.words('english'))]
    if stemming:
        sentence = [PorterStemmer().stem(w) for w in sentence]
    if lemmatization:
        sentence = [WordNetLemmatizer().lemmatize(w) for w in sentence]
    sentence = " ".join(sentence)
    #print(sentence)
    return sentence

def my_lesk(context, lemma):
    """returns word sense for synset found using lesk's algorithm"""
    context = _preprocess(context)#stem, lemmatize, and remove stop words
    #print(context)
    synset = lesk(context, lemma)
    #print(synset)
    if synset:
        return get_sense_key(synset)
    else:
        print('synset empty for {}'.format(lemma))
        return None

def run_lesk_bultin(dev_instances, dev_key):

    predictions = [] 
    targets = []

    for id, wsd in dev_instances.items():
        
        lemma = wsd.lemma.decode("utf-8")
        context = ' '.join([el.decode("utf-8") for el in wsd.context])

        #print('** processing [{}:{}:{}:{}]'.format(id, wsd.index, wsd.lemma, ''.join(context)))

        pred = my_lesk(context, lemma)
        
        predictions.append(pred)
        targets.append(dev_key[id])
   
    #print(predictions[:15])
    #print(targets[:15])
    accuracy = evaluate_accuracy(predictions, targets)

    #print(accuracy)

    return accuracy

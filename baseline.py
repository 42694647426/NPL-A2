from nltk.corpus import wordnet


def get_sense_key(synset):
    """returns the sense key as a string for the given synset"""
    sense_keys = [sense.key() for sense in synset.lemmas()]
    #print(sense_keys)
    return sense_keys

def evaluate_accuracy(predictions, targets):

    correct = 0
    total = len(targets)

    for prediction, target in zip(predictions, targets):
        if set(prediction) & set(target): 
            correct += 1
    accuracy = (correct / total)
    return accuracy

def baseline(lemma):
    """returns most common sense for input wsd.lemma
    """
    synset = wordnet.synsets(lemma)
    if len(synset) > 0:
        return get_sense_key(synset[0])#first one is the most frequent synset. 
    else:
        print('synset empty for {}'.format(lemma))
        return None

def run_baseline(dev_instances, dev_key):
    
    predictions = [] 
    targets = []
    
    for id, wsd in dev_instances.items():
        lemma = wsd.lemma.decode("utf-8")
        pred = baseline(lemma)
        predictions.append(pred)
        targets.append(dev_key[id])
    
    accuracy = evaluate_accuracy(predictions, targets)


    return accuracy
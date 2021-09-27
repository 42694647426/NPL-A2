from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer 
import numpy as np
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import wordnet as wn
import pickle
from nltk import word_tokenize
import string

class MeanVectorizer(object):
    def __init__(self,word2vec):
        self.word2vec=word2vec
        
    def transform(self,X):
        return np.array([
            np.mean([self.word2vec[word.lower()] for word in sentence if word.lower() in self.word2vec.index2word and word.lower() not in stopwords.words('english')]
            or [np.zeros(self.word2vec.vector_size)],axis=0)
            for sentence in X
            ])

def WN_POS(treebank_tag):
    if treebank_tag.islower():
        return treebank_tag
    #nltk.help.upenn_tagset(treebank_tag)
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        # As default pos in lemmatization is Noun
        #print(wn.NOUN)
        return wn.NOUN

def count_lemma(lemma):
    count=0
    if lemma in lemma_sense:
        for key in lemma_sense[lemma]:
            count+=lemma_sense[lemma][key]
    return count
def count_lemma_sense(lemma,senseid):
    count=0
    if lemma in lemma_sense:
        if senseid in lemma_sense[lemma]:
            count=lemma_sense[lemma][senseid]
    return count

def count_sense(lemma):
    return len(wn.synsets(lemma))

def prob_sense_lemma(lemma,senseid):
    c_lem = count_lemma(lemma)
    c_sen = count_sense(lemma)
    c_lem_sen = count_lemma_sense(lemma,senseid)
    if c_lem ==0 and c_sen==0:
        return 0
    else:
        return (c_lem_sen+1)/(c_lem +c_sen)

with open("./semcro_lemma_sense.pkl","rb") as f:
    lemma_sense=pickle.load(f)
'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('stopwords')
'''
def GetTargetWordIndex(W,w):
  TargetWord=[]
  for i in range(len(W)):
    if W[i][0]==w:
      TargetWord.append(i)
  return TargetWord
    
def GetContext(W,i,c):
  left=[]
  index=i
  move=int(c/2)
  while index>0 and (move>0 or c==-1):
    index-=1
    move-=1
    left=[W[index]]+left
  #print("left:",left)
  
  right=[]
  index=i
  move=int(c/2)
  while index<len(W)-1 and (move>0 or c==-1):
    index+=1
    move-=1
    right=right+[W[index]]
  #print("right:",right)
  return left+right

def get_lemma(lemmatizer,text):
    sentence=[]
    for w,p in nltk.pos_tag(nltk.word_tokenize(text)):
        if p!=None:
            sentence.append(lemmatizer.lemmatize(w, pos=WN_POS(p)).lower())
        else:
            sentence.append(lemmatizer.lemmatize(w).lower())
    return sentence

def get_sense_key(synset):
    "returns the sense key as a string for the given synset"
    sense_keys = [sense.key() for sense in synset.lemmas()]
    #print("synset: {}, synset.lemmas: {}".format(synset, synset.lemmas()))
    #print("sensekey: ", sense_keys)
    return sense_keys

def wsdw2v(word2vec,sentence,target_word,context_size=-1,sim_thres=0.37,n_best=1,prob_dist=True):
  # tokenize and lowercase
  lemmatizer = WordNetLemmatizer()
  W = [w for w in nltk.word_tokenize(sentence)]
  # Part-of-Speech Tag
  W = nltk.pos_tag(W)
  # remove stopword
  W = [list(w) for w in W if w[0] not in stopwords.words('english')]
  # lemmatization
  W = [[lemmatizer.lemmatize(w[0],pos=WN_POS(w[1])),w[1]] for w in W]
  #print("Word list:",W)

  # Get target word index
  tw_i=GetTargetWordIndex(W,target_word)
  #print("Target index:",tw_i)
  if tw_i==[]:
    W = [[w[0].lower(), w[1]] for w in W]
    tw_i=GetTargetWordIndex(W,target_word)
    if tw_i == []:
        #print("No target words found! ", target_word)
        return []
  
  for i in tw_i:
    C=GetContext(W,i,context_size)
    C_list=[word[0] for word in C]
    #print("Context:",C_list)
    V_C=MeanVectorizer(model).transform([C_list])
    result=[]
    synset = wn.synsets(W[i][0],WN_POS(W[i][1]))
    if not synset:
        synset = wn.synsets(W[i][0])
        W[i][1]=synset[0].pos()
        #print(W[i][1])

    for synset in wn.synsets(W[i][0],WN_POS(W[i][1])):
      score=0
      #print("{}:{}".format(synset,synset.definition()))
      # sense definition
      Sig=get_lemma(lemmatizer,synset.definition())
      
      # examples
      examples=[]
      for eg in synset.examples():
        examples+=get_lemma(lemmatizer,eg)
      Sig+=examples
      
      # lemmas
      lemmas=[]
      for lm in synset.lemma_names():
        lemmas+=get_lemma(lemmatizer,lm)
      Sig+=list(set(lemmas))
      
      # hyperhypo lemmas
      hyperhyponyms_lemmas=[]
      hyperhyponyms = set(synset.hyponyms() + synset.hypernyms() + synset.instance_hyponyms() + synset.instance_hypernyms())
      for ss in hyperhyponyms:
        for lm in ss.lemma_names():
          hyperhyponyms_lemmas+=get_lemma(lemmatizer,lm)
      Sig+=list(set(hyperhyponyms_lemmas))
      
      #related senses
      related_senses = set(synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms() + \
                                    synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms() + \
                                    synset.similar_tos())
      relatedsenses_lemmas=[]
      for ss in related_senses:
        for lm in ss.lemma_names():
          relatedsenses_lemmas+=get_lemma(lemmatizer,lm)
      Sig+=list(set(relatedsenses_lemmas))
      
      #hypernyms definition
      hypernyms_definitions=[]
      for hyper_synset in synset.hypernyms():
        hypernyms_definitions+=get_lemma(lemmatizer,hyper_synset.definition())
      Sig+=hypernyms_definitions
      
      #hyponyms definition
      hyponyms_definitions=[]
      for hypo_synset in synset.hyponyms():
        hyponyms_definitions+=get_lemma(lemmatizer,hypo_synset.definition())
      Sig+=hyponyms_definitions
      
      #holonyms definition
      holonyms_definitions=[]
      for holo_synset in synset.part_holonyms():
        holonyms_definitions+=get_lemma(lemmatizer,holo_synset.definition())
      
      Sig+=holonyms_definitions
      
      #part meronyms definition
      part_meronyms_definitions=[]
      for mero_synset in synset.part_meronyms():
        part_meronyms_definitions+=get_lemma(lemmatizer,mero_synset.definition())
        
      Sig+=part_meronyms_definitions
      
      #entailment definition
      entailments_definitions=[]
      for entail_synset in synset.entailments():
        entailments_definitions+=get_lemma(lemmatizer,entail_synset.definition())
      Sig+=entailments_definitions
      
      #print("Sig:",Sig)
      V_Sig=MeanVectorizer(model).transform([Sig])
      score=float(cosine_similarity(V_C,V_Sig).flatten())
      method="cosim"
      
      #print(score)
      if prob_dist==True and score<sim_thres:
        first_sense_key=synset.lemmas()[0]._key
        first_sense_key=first_sense_key[first_sense_key.find("%")+1:]
        score=prob_sense_lemma(W[i][0],first_sense_key)
        method="prob_dist"
      result.append({"synset":synset.name(),"score":score,"definition":synset.definition(),"method":method, "key":get_sense_key(synset)})
    result=sorted(result,key=lambda k:k['score'],reverse=True)
    #if result == []:
        #print (W, target_word)
    for j in range(len(result)):
      if j<n_best:
        #print(result[j]["key"])
        return result[j]["key"]
        #print("Score: {}, Synset:{}-{}-{}".format(result[j]["score"],result[j]["synset"],result[j]["definition"],result[j]["key"]))

model = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',binary=True, limit=2000)
#wsdw2v(model,"I go to the bank to deposit money","bank",n_best=5)

# run with our dataset
def evaluate_accuracy(predictions, targets):
    correct = 0
    total = len(targets)
    for prediction, target in zip(predictions, targets):
        #print(type(prediction), type(target))
        if prediction is None or prediction == []:
            #print(target)
            continue
        else:
        #prediction and target are lists, could have more than one answer, so check is 2 list intersect
            if set(prediction) & set(target): 
                correct += 1
    accuracy = (correct / total)
    return accuracy

def run_wsdw2v(dev_instances, dev_key):
    
    model = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',binary=True, limit=2000)
    predictions = [] 
    targets = []

    for id, wsd in dev_instances.items():
        
        lemma = wsd.lemma.decode("utf-8")
        context = ' '.join([el.decode("utf-8") for el in wsd.context])

        #print('** processing [{}:{}:{}:{}]'.format(id, wsd.index, wsd.lemma, ''.join(context)))

        pred = wsdw2v(model,context,lemma,n_best=1)
        
        predictions.append(pred)
        targets.append(dev_key[id])
   
    #print(predictions[:15])
    #print(targets[:15])
    accuracy = evaluate_accuracy(predictions, targets)

    #print(accuracy)
    return accuracy

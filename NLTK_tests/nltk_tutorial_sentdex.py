# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:37:42 2018

@author: Hufsa
"""

import nltk
#nltk.download()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

#ps = PorterStemmer()
#example = "I love Python. I'm doing very pythonic in Pythonly today. How Pythonista are you?"
#
#sentence =[ps.stem(w) for w in word_tokenize(example)]
#print(sentence)
#

from nltk.corpus import state_union # import stateunion addresses
from nltk.tokenize import PunktSentenceTokenizer # unsuperv machine learn tokenizer
# can retrain it if need be

#POS tag list:
#
#CC	coordinating conjunction
#CD	cardinal digit
#DT	determiner
#EX	existential there (like: "there is" ... think of it like "there exists")
#FW	foreign word
#IN	preposition/subordinating conjunction
#JJ	adjective	'big'
#JJR	adjective, comparative	'bigger'
#JJS	adjective, superlative	'biggest'
#LS	list marker	1)
#MD	modal	could, will
#NN	noun, singular 'desk'
#NNS	noun plural	'desks'
#NNP	proper noun, singular	'Harrison'
#NNPS	proper noun, plural	'Americans'
#PDT	predeterminer	'all the kids'
#POS	possessive ending	parent\'s
#PRP	personal pronoun	I, he, she
#PRP$	possessive pronoun	my, his, hers
#RB	adverb	very, silently,
#RBR	adverb, comparative	better
#RBS	adverb, superlative	best
#RP	particle	give up
#TO	to	go 'to' the store.
#UH	interjection	errrrrrrrm
#VB	verb, base form	take
#VBD	verb, past tense	took
#VBG	verb, gerund/present participle	taking
#VBN	verb, past participle	taken
#VBP	verb, sing. present, non-3d	take
#VBZ	verb, 3rd person sing. present	takes
#WDT	wh-determiner	which
#WP	wh-pronoun	who, what
#WP$	possessive wh-pronoun	whose
#WRB	wh-abverb	where, when

sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2005-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
   # print(tokenized[:5]) # printing out first five lines of sample
    try:
        for i in tokenized[:5]: # first 5 lines
            words = nltk.word_tokenize(i) # tokenize by word level, get list
            tagged = nltk.pos_tag(words) # tag sentence part
            chunkGram = r'''Chunk:{<RB.?>*<VB.?>*<NNP>+<NN.*>} 
                                               }<VB.?|IN|DT>+{ '''
            chuckParser = nltk.RegexpParser(chunkGram)
            chunked = chuckParser.parse(tagged)
            chunked.draw()
            
            
    except Exception as e:
        print(str(e))

#process_content()
        
from nltk.stem import WordNetLemmatizer # singular form of words

lemmatizer = WordNetLemmatizer()

#print(lemmatizer.lemmatize("best", pos="a")) # adjective, bc this isn't default noun
#print(lemmatizer.lemmatize("run", pos="v"))
#print(lemmatizer.lemmatize("better", pos="a")) # returns "good"

#print(nltk.__file__) # to find location of nltk to locate corpus
#
from nltk.corpus import gutenberg # call specific corpuse
#from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt") # choose a file
tok = sent_tokenize(sample)

#print(tok[5:15])

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# lemmas = synonyms + related words
#print(syns[0].lemmas()[0].name()) # to get the first word itself

##definition
#print(syns[0].definition())
#
## exammples
#print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms(): # if antonyms exists, lemmas have various methods to retrieve some related word
            antonyms.append(l.antonyms()[0].name())
            
#print(set(synonyms)) # set turns them into dictionary
#print(set(antonyms))

w1 = wordnet.synset("ship.n.01")  # singular synset!
w2 = wordnet.synset("boat.n.01")

# wup compares word similarity
sims = w1.wup_similarity(w2)
print(sims)
    


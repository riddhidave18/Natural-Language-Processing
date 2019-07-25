from __future__ import division
import sys
import math, re
'''
def fetch_words(infile):
    
    pos_words = []
    neg_words = []

    if infile == 'pos-words.txt':
        for words in open(infile):
            pos_words.append(words)
    elif infile == 'neg-words.txt':
        for words in open(infile):
            neg_words.append(words)

    #print "".join(pos_words)
'''
def tokenize(s):
    #tokens = [item.lower() for item in s].split()
	tokens = ''.join(s).lower().split()
	trimmed_tokens = []
	for t in tokens:
		if re.search('\w', t):
			t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
			t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
		trimmed_tokens.append(t)
	return trimmed_tokens

class Naive_bayes:
        global biglist
	global weight
	global labels
	biglist = [] #to store each tokens 
	weight = [] #to know which is higher of that word positive/negative
	labels = [] #whether that word is postive or not
	def __init__(self,textfile,classfile):

                self.pos = dict()
                self.neg = dict()
                self.nutr = dict()
                self.pf = 0
                self.np = 0
                self.nr = 0
		self.train(textfile,classfile)
		#self.classify(devfile)
		
	def train(self,cl,cls):
		#biglist = [] #to store each tokens 
		#weight = [] #to know which is higher of that word positive/negative
		#labels = [] #whether that word is postive or not
	
		classes = []
 
		negs = dict()
		poss = dict()
		neus = dict()
		txtlist = []
		p = 0
		n = 0
		nr = 0
		total_positive = 0
		total_negative = 0
		total_neutral = 0
		vocab = set()


                for tk in cl:
                    txtlist.append(tk)
		
		for i in cls:
		    ####Every line is split as an element in main list### got each wprd from the train classes document
		    classes.append(i)
		for token in tk:
                    wrd = tokenize(token)
		
		for tkt in txtlist:
                        token = tokenize(tkt)
                        
                        
                        for j in token:
                                vocab.add(j)
                                for C in classes:
			#for each token in file check if it is positive  class
                                        if (C == 'positive'):
                                                poss[j] = poss.get(j,0) + 1
                                                p = p + 1
			#or negative
                                        elif (C == 'negative'):
                                                negs[j] = negs.get(j,0) + 1
                                                n = n + 1
			#or neutral
                                        else:	
                                                neus[j] = neus.get(j,0) + 1
                                                nr = nr + 1

                total = p + n + nr
		self.pf = p / total #positive prior
		self.nf = n / total #negative prior
		self.nrf = nr / total #neutral prior
		vocablen = len(vocab)

		for tkt in txtlist:
                        token = tokenize(tkt)

                        for i in token:
                                for c in classes:
                                        if(c == 'positive'):
                                                prob = math.log(((poss.get(c,0)) + 1) / (p + total))
                                                self.pos[c] = self.pos.get(c,prob)
                                        elif (c == 'negative'):
                                                prob = math.log(((negs.get(c,0)) + 1) / (n + total))
                                                self.neg[c] = self.neg.get(c,prob)
                                        else:
                                                prob = math.log(((neus.get(c,0)) + 1) / (nr + total))
                                                self.nutr[c] = self.nutr.get(c,prob)
                                        
                
		
		
##		for index,big in enumerate(biglist):
##			#percentage of that token given positve For eg p['unwanted'|positive]
##			percentage_positive = poss[index] / total_positive
##			#percentage of that token given negative For eg p['unwanted'|negative]
##			percentage_negative = negs[index] / total_negative
##			#percentage of that token given neutral For eg p['unwanted'|neutral]
##			percentage_neutral = neus[index] / total_neutral
##			
##			#naive bayes probability of word with positive For eg p[positive|'unwanted'] = p['unwanted'|positive]*p[positive] ignoring denominator
##			probability_positive = percentage_positive * positive_frac
##			probability_negative = percentage_negative * negative_frac
##			probability_neutral = percentage_neutral * neutral_frac
##			
##			#we consider maximum of both the probability i.e argma	x(p['unwanted'|positive]*p[positive])
##			if(probability_positive > probability_negative):
##				labels.append("positive")
##				weight.append(percentage_positive)
##			elif(probability_negative > probability_positive):
##				labels.append("negative")
##				weight.append(percentage_negative)
##			else:
##				labels.append("neutral")
##				weight.append(percentage_neutral)
		
	    
	def classify(self,testfile):

		
		for t in testfile:
                   textlist.append(t)

                for t in textlist:
                        token = tokenize(t)

                        for sentences in token:
                                if token in self.pos:
                                        positive = self.pf * self.pos[token]
                                if token in self.neg:
                                        negative = self.nf * self.neg[token]
                                if token in self.nutr:
                                        neutral = self.nrt * self.nutr[token]
		

                        maxprob = max(positive, negative, neutral)
                        if maxprob == positive:
                                print 'positive'
                        elif maxprob == negative:
                                print 'negative'
                        else:
                                print 'neutral'


class Binary_Naive_bayes:
        global biglist
	global weight
	global labels
	biglist = [] #to store each tokens 
	weight = [] #to know which is higher of that word positive/negative
	labels = [] #whether that word is postive or not
	def __init__(self,textfile,classfile):

                self.pos = dict()
                self.neg = dict()
                self.nutr = dict()
                self.pf = 0
                self.np = 0
                self.nr = 0
		self.train(textfile,classfile)
		#self.classify(devfile)
		
	def train(self,cl,cls):
		#biglist = [] #to store each tokens 
		#weight = [] #to know which is higher of that word positive/negative
		#labels = [] #whether that word is postive or not
	
		classes = []
 
		negs = dict()
		poss = dict()
		neus = dict()
		txtlist = []
		p = 0
		n = 0
		nr = 0
		total_positive = 0
		total_negative = 0
		total_neutral = 0
		vocab = set()


                for tk in cl:
                    txtlist.append(tk)
		
		for i in cls:
		    ####Every line is split as an element in main list### got each wprd from the train classes document
		    classes.append(i)
		for token in tk:
                    wrd = tokenize(token)
		
		for tkt in txtlist:
                        token = tokenize(tkt)
                        
                        
                        for j in token:
                                vocab.add(j)
                                for C in classes:
			#for each token in file check if it is positive  class
                                        if (C == 'positive'):
                                                poss[j] = poss.get(j,0) + 1
                                                p = p + 1
			#or negative
                                        elif (C == 'negative'):
                                                negs[j] = negs.get(j,0) + 1
                                                n = n + 1
			#or neutral
                                        else:	
                                                neus[j] = neus.get(j,0) + 1
                                                nr = nr + 1

                total = p + n + nr
		self.pf = p / total #positive prior
		self.nf = n / total #negative prior
		self.nrf = nr / total #neutral prior
		vocablen = len(vocab)

		for tkt in txtlist:
                        token = tokenize(tkt)

                        for i in token:
                                for c in classes:
                                        if(c == 'positive'):
                                                prob = math.log(((poss.get(c,0)) + 1) / (p + total))
                                                self.pos[c] = self.pos.get(c,prob)
                                        elif (c == 'negative'):
                                                prob = math.log(((negs.get(c,0)) + 1) / (n + total))
                                                self.neg[c] = self.neg.get(c,prob)
                                        else:
                                                prob = math.log(((neus.get(c,0)) + 1) / (nr + total))
                                                self.nutr[c] = self.nutr.get(c,prob)
                                        
                
		
		
##		for index,big in enumerate(biglist):
##			#percentage of that token given positve For eg p['unwanted'|positive]
##			percentage_positive = poss[index] / total_positive
##			#percentage of that token given negative For eg p['unwanted'|negative]
##			percentage_negative = negs[index] / total_negative
##			#percentage of that token given neutral For eg p['unwanted'|neutral]
##			percentage_neutral = neus[index] / total_neutral
##			
##			#naive bayes probability of word with positive For eg p[positive|'unwanted'] = p['unwanted'|positive]*p[positive] ignoring denominator
##			probability_positive = percentage_positive * positive_frac
##			probability_negative = percentage_negative * negative_frac
##			probability_neutral = percentage_neutral * neutral_frac
##			
##			#we consider maximum of both the probability i.e argma	x(p['unwanted'|positive]*p[positive])
##			if(probability_positive > probability_negative):
##				labels.append("positive")
##				weight.append(percentage_positive)
##			elif(probability_negative > probability_positive):
##				labels.append("negative")
##				weight.append(percentage_negative)
##			else:
##				labels.append("neutral")
##				weight.append(percentage_neutral)
		
	    
	def classify(self,testfile):

		
		for t in testfile:
                   textlist.append(t)

                for t in textlist:
                        token = tokenize(t)

                        for sentences in token:
                                if token in self.pos:
                                        positive = self.pf * self.pos[token]
                                if token in self.neg:
                                        negative = self.nf * self.neg[token]
                                if token in self.nutr:
                                        neutral = self.nrt * self.nutr[token]
		

                        maxprob = max(positive, negative, neutral)
                        if maxprob == positive:
                                print 'positive'
                        elif maxprob == negative:
                                print 'negative'
                        else:
                                print 'neutral'

class Polarity:
    def __init__(self,test_texts,pos_texts,neg_texts):
        self.compare(test_texts,pos_texts,neg_texts)

    def compare(self,test_texts,pos_texts,neg_texts):
        pos_words = dict()
        neg_words = dict()
        data = dict()

        dev_list = []
        for word in pos_texts:
            pos_words[word] = pos_words.get(word,0) 
        for word in neg_texts:
            neg_words[word] = neg_words.get(word,0)
        for data in test_texts:
            dev_list.append(data)
            for data in dev_list:
                token = tokenize(data)
                positve_count = 0
                negative_count = 0
                for t in token:
                    if t in pos_words:
                        positve_count = positve_count + 1
                    elif t in neg_words:
                        negative_count = negative_count + 1

            if(positve_count > negative_count):
                print 'positive'
            elif(positve_count < negative_count):
                print 'negative'
            else:
                print 'neutral'


		
if __name__ == '__main__':

    method = sys.argv[1]
    train_texts_fname = sys.argv[2]
    train_classes_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    train_texts = [x.strip() for x in open(train_texts_fname)]
    train_classes = [x.strip() for x in open(train_classes_fname)]
    test_texts = [x.strip() for x in open(test_texts_fname)]
	
        if method == 'nb':
		test_sents = Naive_bayes(train_texts,train_classes)
		test_sents.classify(test_texts)
		#test_sents = Naive_bayes.classify(test_texts)
		#test_sents = [classify(x) for x in test_texts]
        elif method == 'nbbin':
                test_sents = Naive_bayes(train_texts,train_classes)
		test_sents.classify(test_texts)
	elif method == 'lexicon':
                pos_words = [x.strip() for x in open('pos-words.txt')]
                neg_words = [x.strip() for x in open('neg-words.txt')]
                classifier = Polarity(test_texts,pos_words,neg_words)
		

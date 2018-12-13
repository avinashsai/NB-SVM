import sys
import os
import re
import numpy as np 
from sklearn.utils import shuffle
from nltk.corpus import stopwords

stopword = stopwords.words('english')

def pre_process(text):
	text = re.sub(r"it\'s","it is",str(text))
	text = re.sub(r"i\'d","i would",str(text))
	text = re.sub(r"don\'t","do not",str(text))
	text = re.sub(r"he\'s","he is",str(text))
	text = re.sub(r"there\'s","there is",str(text))
	text = re.sub(r"that\'s","that is",str(text))
	text = re.sub(r"can\'t", "can not", text)
	text = re.sub(r"cannot", "can not ", text)
	text = re.sub(r"what\'s", "what is", text)
	text = re.sub(r"What\'s", "what is", text)
	text = re.sub(r"\'ve ", " have ", text)
	text = re.sub(r"n\'t", " not ", text)
	text = re.sub(r"i\'m", "i am ", text)
	text = re.sub(r"I\'m", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"\'s"," is",text)
	text = re.sub(r"[^a-zA-Z0-9]"," ",str(text))
	words = text.split()

	return " ".join(word.lower() for word in words if word.lower() not in stopword)



def load_rtpoldata(path):
	corpus = []
	poscount = 0
	negcount = 0
	with open(path+'rt-polarity.pos','r',encoding='latin1') as f:
		for line in f.readlines():
			poscount+=1
			corpus.append(pre_process(line[:-1]))

	with open(path+'rt-polarity.neg','r',encoding='latin1') as f:
		for line in f.readlines():
			negcount+=1
			corpus.append(pre_process(line[:-1]))

	labels = np.zeros(len(corpus))
	labels[0:5331] = 1
	bias = np.log(poscount/negcount)


	return corpus,labels,bias




def load_subobjdata(path):
	corpus = []
	poscount = 0
	negcount = 0
	with open(path+'sub.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			poscount+=1
			corpus.append(pre_process(line[:-1]))

	with open(path+'obj.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			negcount+=1
			corpus.append(pre_process(line[:-1]))

	labels = np.zeros(len(corpus))
	labels[0:5000] = 1

	bias = np.log(poscount/negcount)

	return corpus,labels,bias


def load_imdb(path):
	traincorpus = []
	testcorpus = []
	poscount = 0
	negcount= 0
	with open(path+'train_pos.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			poscount+=1
			traincorpus.append(pre_process(line[:-1]))

	with open(path+'train_neg.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			negcount+=1
			traincorpus.append(pre_process(line[:-1]))

	with open(path+'test_pos.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			poscount+=1
			testcorpus.append(pre_process(line[:-1]))

	with open(path+'test_neg.txt','r',encoding='latin1') as f:
		for line in f.readlines():
			negcount+=1
			testcorpus.append(pre_process(line[:-1]))

	trainlabels = np.zeros(25000)
	trainlabels[0:12500] = 1

	testlabels = np.zeros(25000)
	testlabels[0:12500] = 1

	bias = np.log(poscount/negcount)

	return traincorpus,testcorpus,trainlabels,testlabels,bias





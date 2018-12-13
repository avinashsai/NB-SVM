import sys
import os
import numpy as np 
import scipy as sp 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


svm_c = 0.1
nbsvm_c = 1
alpha = 1
beta = 0.25

def get_features(Xtrain,Xtest,method,ngram=1):
	if(method=='tfidf'):
		vec = TfidfVectorizer(ngram_range=(1,ngram),min_df=2,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')
	else:
		vec = CountVectorizer(ngram_range=(1,ngram),min_df=2,max_df=0.8,stop_words='english')
	train_features = vec.fit_transform(Xtrain)
	test_features = vec.transform(Xtest)

	return train_features,test_features,len(vec.vocabulary_)


def calculate_r(Xtrain,ytrain):
	trainsize = Xtrain.shape[0]
	feasize = Xtrain.shape[1]
	p = np.ones(feasize) * alpha
	q = np.ones(feasize) * alpha

	for i in range(trainsize):
		if(ytrain[i]):
			p+=Xtrain[i,:]
		else:
			q+=Xtrain[i,:]

	p_counts = p/abs(p).sum()
	q_counts = q/abs(q).sum()

	return np.log((p_counts/q_counts))


def get_nbsvmfeatures(Xtrain,Xtest,rval):
	trainsize = Xtrain.shape[0]
	testsize = Xtest.shape[0]
	feasize = Xtrain.shape[1]

	rv = rval.reshape((1,feasize))
	train_nbsvm = np.zeros((trainsize,feasize))
	test_nbsvm = np.zeros((testsize,feasize))

	for i in range(0,trainsize):
		indices = np.nonzero(Xtrain[i,:])
		for index in indices:
			train_nbsvm[i,index] = rv[0,index]


	for i in range(0,testsize):
		indices = np.nonzero(Xtest[i,:])
		for index in indices:
			test_nbsvm[i,index] = rv[0,index]

	return train_nbsvm,test_nbsvm

def cal_testacc(Xtest,ytest,svcoeff,bias):
	correct = 0
	for i in range(Xtest.shape[0]):
		sig = np.sign((np.dot(svcoeff,Xtest[i,:].T)+bias))
		if(sig>0):
			label = 1
		else:
			label = 0
		if(label==ytest[i]):
			correct+=1
	return correct

def nbsvm_kfold(data,labels,method,ngramrange,bias):
	kf = StratifiedKFold(n_splits=10)

	mnb_acc = 0
	svm_acc = 0
	nbsvm_acc = 0

	total_len = len(data)
	for trainindex,testindex in kf.split(data,labels):
		traincorpus = [data[index] for index in trainindex]
		testcorpus = [data[index] for index in testindex]
		trainlabels = labels[trainindex]
		testlabels = labels[testindex]

		trainfea,testfea,cur_size = get_features(traincorpus,testcorpus,method,ngramrange)


		mnb = MultinomialNB()
		mnb.fit(trainfea,trainlabels)
		mnb_pred = mnb.predict(testfea)
		mnb_acc+=np.sum(mnb_pred==testlabels)


		sv = LinearSVC(C=svm_c)
		sv.fit(trainfea,trainlabels)
		sv_coeff = sv.coef_

		if(bias==0):
			sv_pred = sv.predict(testfea)
			svm_acc+=np.sum(sv_pred==testlabels)
		else:
			svm_acc+=cal_testacc(testfea,testlabels,sv_coeff,bias)

		rval = calculate_r(trainfea,trainlabels)

		nbsvm_train,nbsvm_test = get_nbsvmfeatures(trainfea,testfea,rval)

		nbsvm = LinearSVC(C=nbsvm_c)
		nbsvm.fit(nbsvm_train,trainlabels)

		nbsvm_coeff = nbsvm.coef_

		w_bar = abs(nbsvm_coeff).sum()/cur_size
		w_dash = (1-beta)*w_bar + beta * nbsvm_coeff

		nbsvm_acc+=cal_testacc(nbsvm_test,testlabels,w_dash,bias)


	print("MNB Accuracy {}".format((mnb_acc/total_len)*100))
	print("SVM Accuracy {}".format((svm_acc/total_len)*100))
	print("NBSVM Accuracy {}".format((nbsvm_acc/total_len)*100))



def nbsvm_holdout(train_data,test_data,train_labels,test_labels,method,ngramrange,bias):

	trainfea,testfea,cur_size = get_features(train_data,test_data,method,ngramrange)

	testlength = len(test_data)

	mnb = MultinomialNB()
	mnb.fit(trainfea,train_labels)
	mnb_pred = mnb.predict(testfea)
	mnb_acc=accuracy_score(mnb_pred,test_labels)


	sv = LinearSVC(C=svm_c)
	sv.fit(trainfea,train_labels)
	sv_coeff = sv.coef_
	if(bias==0):
		sv_pred = sv.predict(testfea)
		svm_acc=accuracy_score(sv_pred,test_labels)
	else:
		svm_acc=cal_testacc(testfea,test_labels,sv_coeff,bias)
		svm_acc = (svm_acc/testlength)

	rval = calculate_r(trainfea,train_labels)

	nbsvm_train,nbsvm_test = get_nbsvmfeatures(trainfea,testfea,rval)

	nbsvm = LinearSVC(C=nbsvm_c)
	nbsvm.fit(nbsvm_train,train_labels)

	nbsvm_coeff = nbsvm.coef_

	w_bar = abs(nbsvm_coeff).sum()/cur_size
	w_dash = (1-beta)*w_bar + beta * nbsvm_coeff

	nbsvm_acc=cal_testacc(nbsvm_test,test_labels,w_dash,bias)
	nbsvm_acc = (nbsvm_acc/testlength)


	print("MNB Accuracy {}".format(mnb_acc*100))
	print("SVM Accuracy {}".format(svm_acc*100))
	print("NBSVM Accuracy {}".format(nbsvm_acc*100))

import os
import sys
import re
import numpy as np 
import pandas as pd 
import argparse

from loader import *
from nbsvm import *
from sklearn.utils import shuffle

rt_polarity_path = '../rt-polaritydata/'
sub_obj_path = '../Sub_Obj/'
imdb_path = '../IMDB/'

def main():
	dataset = sys.argv[1]
	method = sys.argv[2]
	ngramrange = int(sys.argv[3])

	if(dataset=='rt-polarity'):
		data,labels,bias = load_rtpoldata(rt_polarity_path)
		nbsvm_kfold(data,labels,method,ngramrange,bias)

	elif(dataset=='subj_obj'):
		data,labels,bias = load_subobjdata(sub_obj_path)
		nbsvm_kfold(data,labels,method,ngramrange,bias)
		
	elif(dataset=='IMDB'):
		traindata,testdata,trainlabels,testlabels,bias = load_imdb(imdb_path)
		traindata,trainlabels = shuffle(traindata,trainlabels)
		testdata,testlabels = shuffle(testdata,testlabels)
		nbsvm_holdout(traindata,testdata,trainlabels,testlabels,method,ngramrange,bias)


if __name__ == '__main__':
	main()

# Getting Started
This is the implementation of the paper Baselines and Bigrams: Simple, Good Sentiment and Topic Classification https://www.aclweb.org/anthology/P12-2018. 

# Implementation Details
I have implemented the paper for the following datasets

```
1. sentence polarity dataset 2.0(rt-polaritydata Folder)

2. subjectivity dataset v1.0(Sub_Obj Folder)

3. Large Movie Review Dataset v1.0(IMDB Folder)

```
The datasets can be found here: http://ai.stanford.edu/~amaas/data/sentiment/ and http://www.cs.cornell.edu/people/pabo/movie-review-data/


# Additional Features Implemented

Addition to Binarized count based features I have implemented TF-IDF also.

# Packages Required

```
1. python>=3.5
2. Numpy
3. sklearn
4. NLTK

```

# How to run

1. Clone the repository using:

```
git clone https://github.com/avinashsai/NB-SVM.git

```
 2. Get the results for the desired dataset,desired method, desired number of ngrams using:
 
 ```
 python main.py <dataset-name> <method> <ngram-count>
 
 dataset-name includes:
 
 rt-polarity
 
 subj_obj
 
 IMDB
 
method includes:

tfidf

count

ngram-count includes:

1

2

```

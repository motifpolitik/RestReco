"""
Script:     05-Tune-Linear-SVM.py
Purpose:    Hypertune Linear SVM (Best Performing Classifer) & Output Final Classifiers;
			Print Final Classifier Statistics // TODO: Build hyperparemeter tuning into model selection from (04)
Input:      data/SemEval/all_semeval_data.pkl
            data/vectorizers/binary_vectorizer.pkl
Output:     data/classifiers/lsvm_food_positive.pkl
            data/classifiers/lsvm_food_negative.pkl
	    data/classifiers/lsvm_service_positive.pkl
	    data/classifiers/lsvm_service_negative.pkl
	    data/classifiers/lsvm_ambience_positive.pkl
	    data/classifiers/lsvm_ambience_negative.pkl
	    data/classifiers/lsvm_value_positive.pkl
	    data/classifiers/lsvm_value_negative.pkl
"""
import warnings
import numpy as np
import pandas as pd
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore")

##Read in Binary Text Vectorizer
with open("data/vectorizers/binary_vectorizer.pkl",'rb') as f:
    binary_vectorizer = pickle.load(f)
    

##Read In and Clean ABSA Data - Create Binary Flags for Topics & Aggregate to Sentence Level
semeval_data = pd.read_pickle("data/SemEval/all_semeval_data.pkl")

food = ["food","FOOD#QUALITY","FOOD#STYLE_OPTIONS","DRINKS#QUALITY","DRINKS#STYLE_OPTIONS","FOOD#GENERAL"]
service = ["service","SERVICE#GENERAL"]
ambience = ["ambience","AMBIENCE#GENERAL"]
value = ["price","FOOD#PRICES","RESTAURANT#PRICES","DRINKS#PRICES"]

aspects = ["topic_food_positive","topic_food_negative","topic_service_positive","topic_service_negative","topic_ambience_positive","topic_ambience_negative","topic_value_positive","topic_value_negative"]

semeval_data["topic_food_positive"] = 0
semeval_data["topic_food_negative"] = 0
semeval_data["topic_service_positive"] = 0
semeval_data["topic_service_negative"] = 0
semeval_data["topic_ambience_positive"] = 0
semeval_data["topic_ambience_negative"] = 0 
semeval_data["topic_value_positive"] = 0
semeval_data["topic_value_negative"] = 0

semeval_data.ix[((semeval_data.category.isin(food)) & (semeval_data.polarity == 'positive')), "topic_food_positive"] = 1
semeval_data.ix[((semeval_data.category.isin(food)) & (semeval_data.polarity == 'negative')), "topic_food_negative"] = 1
semeval_data.ix[((semeval_data.category.isin(service)) & (semeval_data.polarity == 'positive')), "topic_service_positive"] = 1
semeval_data.ix[((semeval_data.category.isin(service)) & (semeval_data.polarity == 'negative')), "topic_service_negative"] = 1
semeval_data.ix[((semeval_data.category.isin(ambience)) & (semeval_data.polarity == 'positive')), "topic_ambience_positive"] = 1
semeval_data.ix[((semeval_data.category.isin(ambience)) & (semeval_data.polarity == 'negative')), "topic_ambience_negative"] = 1
semeval_data.ix[((semeval_data.category.isin(value)) & (semeval_data.polarity == 'positive')), "topic_value_positive"] = 1
semeval_data.ix[((semeval_data.category.isin(value)) & (semeval_data.polarity == 'negative')), "topic_value_negative"] = 1

semeval_data = semeval_data.groupby(by="sentence", as_index=False)[aspects].max()


##Test Train Split
semeval_train, semeval_test = train_test_split(semeval_data, test_size=0.25, random_state=4444)



"""
Function to Hypertune Linear SVM Parameters on Training Set (optimized for recall)

vectorizer:		Vectorizer to use in model training
train:			Training data (pandas DF)
topic:			Topic to train model to identify (food, service, ambience, value)

RETURNS:		Hyptertuned linear SVM classifier for given topic
"""
def tune_linear_svm(vectorizer, train, topic):
	grid={"C": [0.05,0.5,1,1.5,2,5,10], "loss": ["hinge", "squared_hinge"], "class_weight": [None,"balanced"]}

	train_X = vectorizer.transform(train["sentence"]).toarray()
	train_y = train[topic]

	clf = GridSearchCV(LinearSVC(), grid, scoring="recall", cv=3).fit(train_X, train_y)

	print("Best Params for "+topic+":", clf.best_params_)
	print("Best Score: "+str(clf.best_score_))
	print("")

	return clf


"""
Function to Evaluate Final SVM Parameters on Test Set

vectorizer:		Vectorizer used in model training
test:			Test data (pandas DF)
clf:			Trained text classifer for given topic
topic:			Topic model trained to identify
RETURNS:		Test data statistics (Accuracy, precision, recall, confusion matrix)
"""
def final_test_stats(vectorizer, test, clf, topic):
	test_X = vectorizer.transform(test["sentence"]).toarray()
	test_y = test[topic]

	print("Final Statistics for ", topic)
	print("Accuracy:  ", accuracy_score(test_y, clf.predict(test_X)))
	print("Precision: ", precision_score(test_y, clf.predict(test_X)))
	print("Recall:    ", recall_score(test_y, clf.predict(test_X)))
	print("Confusion Matrix: ")
	print(confusion_matrix(test_y, clf.predict(test_X)))
	print("")


"""
Function to Pickle Classifiers for Future Analysis

clf:		Classifer to be pickled
outfile:	File path to save classifer to
RETURNS:	Pickled classifier
"""
def pickle_clf(clf, outpickle):
	with open(outpickle, "wb") as f:
		pickle.dump(clf, f)


##Use Above Functions to Hypertune Model
print("Grid Searching for Optimal Parameters...")
lsvm_food_positive = tune_linear_svm(binary_vectorizer, semeval_train, "topic_food_positive")
lsvm_food_negative = tune_linear_svm(binary_vectorizer, semeval_train, "topic_food_negative") 
lsvm_service_positive = tune_linear_svm(binary_vectorizer, semeval_train, "topic_service_positive")
lsvm_service_negative = tune_linear_svm(binary_vectorizer, semeval_train, "topic_service_negative")
lsvm_ambience_positive = tune_linear_svm(binary_vectorizer, semeval_train, "topic_ambience_positive")
lsvm_ambience_negative = tune_linear_svm(binary_vectorizer, semeval_train, "topic_ambience_negative")
lsvm_value_positive = tune_linear_svm(binary_vectorizer, semeval_train, "topic_value_positive")
lsvm_value_negative = tune_linear_svm(binary_vectorizer, semeval_train, "topic_value_negative")
print("")


##Print Final Model Statistics:
print("Tuned Linear SVM Statistics...")
final_test_stats(binary_vectorizer, semeval_test, lsvm_food_positive, "topic_food_positive")
final_test_stats(binary_vectorizer, semeval_test, lsvm_food_negative, "topic_food_negative")
final_test_stats(binary_vectorizer, semeval_test, lsvm_service_positive, "topic_service_positive")
final_test_stats(binary_vectorizer, semeval_test, lsvm_service_negative, "topic_service_negative")
final_test_stats(binary_vectorizer, semeval_test, lsvm_ambience_positive, "topic_ambience_positive")
final_test_stats(binary_vectorizer, semeval_test, lsvm_ambience_negative, "topic_ambience_negative")
final_test_stats(binary_vectorizer, semeval_test, lsvm_value_positive, "topic_value_positive")
final_test_stats(binary_vectorizer, semeval_test, lsvm_value_negative, "topic_value_negative")


#Pickle Clasifiers
print("Pickling Classifiers...")
pickle_clf(lsvm_food_positive, "data/classifiers/lsvm_food_positive.pkl")
pickle_clf(lsvm_food_negative, "data/classifiers/lsvm_food_negative.pkl")
pickle_clf(lsvm_service_positive, "data/classifiers/lsvm_service_positive.pkl")
pickle_clf(lsvm_service_negative, "data/classifiers/lsvm_service_negative.pkl")
pickle_clf(lsvm_ambience_positive, "data/classifiers/lsvm_ambience_positive.pkl")
pickle_clf(lsvm_ambience_negative, "data/classifiers/lsvm_ambience_negative.pkl")
pickle_clf(lsvm_value_positive, "data/classifiers/lsvm_value_positive.pkl")
pickle_clf(lsvm_value_negative, "data/classifiers/lsvm_value_negative.pkl")


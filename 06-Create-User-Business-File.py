
"""
Script:     06-Create-User-Business-File.py
Purpose:    Classify each Yelp Review sentence by topic and aggregate to User and Business level     
Input:      data/vectorizers/binary_vectorizer.pkl (pickled sklearn vectorizer object)
            data/classifiers/lsvm_food_positive.pkl
            data/classifiers/lsvm_food_negative.pkl
            data/classifiers/lsvm_service_positive.pkl
            data/classifiers/lsvm_service_negative.pkl
            data/classifiers/lsvm_ambience_positive.pkl
            data/classifiers/lsvm_ambience_negative.pkl
            data/classifiers/lsvm_value_positive.pkl
            data/classifiers/lsvm_value_negative.pkl (pickled sklean classifier objects)
            data/yelp/dataframes/review_sentences_Carolina.pkl
            data/yelp/dataframes/review_sentences_PA.pkl
            data/yelp/dataframes/review_sentences_WI.pkl (Yelp Review Data as pickled pandas DFs)
Output:     data/yelp/dataframes/yelp_review_user.pkl (User-level data as pickled pandas DF)
"""

import warnings
import numpy as np
import pandas as pd
import pickle

from sklearn.manifold import TSNE
np.set_printoptions(suppress=True)



"""
Function to Read in Pickled Classifiers 

path:       Path to pickled classifier
RETURNS:    Deserialized classifer
"""
def open_pickle(path):
    with open(path,'rb') as f:
        out = pickle.load(f)
    return out


"""
Function to Stack Yelp data from multiple cities into single pandas DF

base:       Previous Yelp Dataset
inpath:     New Yelp data to add to corpus
RETURNS:    Yelp Dataset with new data added
"""
def read_in_yelp(base, inpath):
    df = pd.read_pickle(inpath)
    return pd.concat([base, df], axis=0)


"""
Function to Classify Yelp Review Sentences by Topic

vectorizer:     Vectorizer used to train text classifier
df:             Yelp review data, split out by sentence
clf:            Classifier to predict sentence topics
topic:          Topic classifier is attempting to predict
RETURNS:        Yelp review with prediction as to whether sentence discusses given topic (0/1)
"""
def classify_sentences(vectorizer, df, clf, topic):
    X = vectorizer.transform(df["sentence"]) #Transform Yelp Data onto Word Vector Space
    p = pd.Series(clf.predict(X), name=topic)

    return pd.concat([df, p], axis=1)

"""
Function aggregate topic tagged sentences to user level 

df:     Yelp sentence review-level data
RETURNS:    Yelp user-level dataset, reporting proportion of user's sentences discussing each topic
"""
def create_user_file(df):
    return df.groupby(by="user_id", as_index=False)["topic_food_positive","topic_food_negative","topic_service_positive","topic_service_negative","topic_ambience_positive","topic_ambience_negative","topic_value_positive","topic_value_negative","relevant","total"].sum()


"""
Function aggregate topic tagged sentences to business level 

df:     Yelp sentence review-level data
RETURNS:    Yelp business-level dataset, reporting proportion of business's sentences discussing each topic
"""
def create_business_file(df):
    return df.groupby(by="business_id", as_index=False)["topic_food_positive","topic_food_negative","topic_service_positive","topic_service_negative","topic_ambience_positive","topic_ambience_negative","topic_value_positive","topic_value_negative","relevant","total"].sum()


"""
Function for mapping user to businesses reviewed 

df:     Yelp sentence review-level data
RETURNS:    Yelp business-level dataset, reporting proportion of business's sentences discussing each topic
"""
def create_user_business_file(df):
    # return df.groupby(['user_id','business_id'])["rating"].unique()
    return df.groupby(['user_id', 'business_id'])['rating'].unique().reset_index(name='rating')



"""
Function to pickle user-level Yelp data for further analysis

item:   Dataset to by pickled
outpath:    Path to pickle dataset at
RETURNS:    Pickled dataset
"""
def save_pickle(item, outpath):
    with open(outpath, "wb") as f:
        pickle.dump(item, f)


warnings.filterwarnings("ignore")

##Run Functions
print("Reading In Vectorizer and Classifiers...")
binary_vectorizer = open_pickle("data/vectorizers/binary_vectorizer.pkl")

lsvm_food_positive = open_pickle("data/classifiers/lsvm_food_positive.pkl")
lsvm_food_negative = open_pickle("data/classifiers/lsvm_food_negative.pkl")
lsvm_service_positive = open_pickle("data/classifiers/lsvm_service_positive.pkl")
lsvm_service_negative = open_pickle("data/classifiers/lsvm_service_negative.pkl")
lsvm_ambience_positive = open_pickle("data/classifiers/lsvm_ambience_positive.pkl")
lsvm_ambience_negative = open_pickle("data/classifiers/lsvm_ambience_negative.pkl")
lsvm_value_positive = open_pickle("data/classifiers/lsvm_value_positive.pkl")
lsvm_value_negative = open_pickle("data/classifiers/lsvm_value_negative.pkl")



##Initalize Container for Yelp Data (Pandas DF)
print("Reading in Yelp Review Data...")
yelp = pd.DataFrame()
yelp = read_in_yelp(yelp, "data/yelp/dataframes/review_sentences_Carolina.pkl")
yelp = read_in_yelp(yelp, "data/yelp/dataframes/review_sentences_PA.pkl")
yelp = read_in_yelp(yelp, "data/yelp/dataframes/review_sentences_WI.pkl")
yelp.reset_index(inplace=True, drop=True)                                   #Reset Index after Stacking

print("Total Number of Yelp Users:     ", yelp["user_id"].nunique())         #Print Number of Users in Yelp Masterfile
print("Total Number of Yelp Reviews:   ", yelp["review_id"].nunique())      #Print Number of Reviews in Yelp Masterfile
print("Total Number of Yelp Businesses ", yelp["business_id"].nunique())      #Print Number of Businesses in Yelp Masterfil
print("Total Number of Yelp Sentences: ", yelp.shape[0])                  #Print Number of Sentences in Yelp Masterfile


##Classify Sentences
print("Classifying Yelp Review Data...")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_food_positive, "topic_food_positive")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_food_negative, "topic_food_negative")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_service_positive, "topic_service_positive")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_service_negative, "topic_service_negative")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_ambience_positive, "topic_ambience_positive")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_ambience_negative, "topic_ambience_negative")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_value_positive, "topic_value_positive")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_value_negative, "topic_value_negative")

yelp["relevant"] = yelp[["topic_food_positive", "topic_food_negative", "topic_service_positive", "topic_service_negative", "topic_ambience_positive", "topic_ambience_positive", "topic_ambience_negative", "topic_value_positive", "topic_value_negative"]].max(axis=1)   #Create Flag for Topic-Relevant Sentences (total)
yelp["total"] = 1


##Sum Up to User Level
print("Aggregating to User Level...")
user = create_user_file(yelp)
#print("Size of User-Level File: ", user.shape[0])

##Sum Up to Business Level
print("Aggregating to Business Level...")
business = create_business_file(yelp)
#print("Size of Business-Level File: ", business.shape[0])

##Create User-Business Mapping
print("Aggregating Businesses for each user...")
user_business = create_user_business_file(yelp)
#print("Size of User Business-Level File: ", user_business.shape[0])


#Pickle User Level File
print("Pickling User-Level File...")
save_pickle(user, "data/yelp/dataframes/yelp_review_user.pkl")

#Pickle Business Level File
print("Pickling Business-Level File...")
save_pickle(business, "data/yelp/dataframes/yelp_review_business.pkl")

#Pickle User-Business Level File
print("Pickling User-Business-Level File...")
save_pickle(user_business, "data/yelp/dataframes/yelp_review_user_business.pkl")

# Restaurant Recommender System
A system for recommending restaurants from their online reviews. This is built on the topic modeling system described in the blog, [yelp me out](http://www.huguedata.com/2016/07/15/yelp-me-out/). 


### Code Description

The table below provides high-level overviews of what each analysis script does. More information (including specific input/ouput data) can be found in each script's header.

Program 	| Description | 
----------- | ----------- |
01-Preprocess-Yelp-Data.py | Clean raw Yelp data and split into sentences so that 1 record = 1 review sentence.
02-Parse-SemEval-Data.py | Parse raw SemEval data (XML format) into pandas dataframe.
03-Vectorizers.py | Fit binary, count, and TF-IDF vectorizers to entire vocubulary from SemEval and Yelp data.
04-Train-Classifiers.py | Train text classifiers on SemEval data and test performance.
05-Tune-Classifiers.py | Tune hyperparameters for best performing text classifier from (4).
06-Create-User-Business-File.py | Predict aspects in Yelp review sentences and aggregate to the user and business level.
07-Yelp-Rating-Reco.ipnyb | Restaurant recommendations based on aspects and user preferences along with the user ratings
07-Yelp-Cluster.ipynb | Restaurant recommendations based on aspects and user preferences


### Dependencies
In order to run the scripts described above, you will need the following Python libraries installed: 

Numpy, Pandas, Sci-Kit Learn, Matplotlib, Seaborn, PyLab, Pickle
Xml, Json, Collections, NLTK, MongoDB and PyMongo, Warnings.

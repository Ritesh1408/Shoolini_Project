# Shoolini_Project
Twitter Pfizer project analysis using Machine Learning

This project walks you on how to create a twitter sentiment analysis model using python. Twitter sentiment analysis is performed to identify the sentiments of the people towards various topics. For this project, we will be analysing the sentiment of people towards Pfizer vaccines. We will be using the data available on Kaggle to create this machine learning model. The collected tweets from Twitter will be analysed using machine learning to identify the different sentiments present in the tweets. The different sentiments identified in this project include positive sentiment, negative sentiment and neutral sentiment. We will also be using different classifiers to see which classifier gives the best model accuracy.


### Sentiment Analysis on Pfizer Vaccine Tweets

#### Overview:
This project focuses on analyzing sentiments expressed in tweets related to Pfizer vaccines using machine learning techniques. It aims to understand public perception and sentiment towards Pfizer vaccines through the analysis of Twitter data.

#### Dataset:
The dataset used in this project is sourced from Kaggle, containing tweets discussing Pfizer vaccines. It includes various features such as tweet text, user information, date, retweet counts, and favorites.

#### Libraries and Tools:
- pandas: For data manipulation and analysis.
- numpy: For numerical computing.
- matplotlib and seaborn: For data visualization.
- nltk: For natural language processing tasks such as tokenization and stopwords removal.
- WordCloud: For generating word clouds to visualize frequent words.
- scikit-learn: For machine learning tasks including text vectorization and model training.
- warnings: For suppressing specific warning messages during analysis.

#### Preprocessing:
- Removed irrelevant features such as user-related information and metadata.
- Cleaned and processed the tweet text using regex for removing URLs, usernames, and special characters.
- Tokenized the text, removed stopwords, and converted it to lowercase.

#### Sentiment Analysis:
- Utilized TextBlob library to compute the polarity of each tweet, indicating positive, negative, or neutral sentiment.
- Categorized tweets into sentiment classes based on polarity scores.
- Visualized the distribution of sentiment classes using count plots.

#### Word Cloud Visualization:
- Generated word clouds to visualize the most frequent words in positive, negative, and neutral tweets.

#### Model Training and Evaluation:
- Utilized the Bag-of-Words approach for text vectorization.
- Split the dataset into training and testing sets.
- Trained a Support Vector Machine (SVM) classifier on the training data.
- Evaluated the model's performance using accuracy score, F1-score, and confusion matrix.
- Conducted hyperparameter tuning using RandomizedSearchCV to optimize the SVM model.

#### Results:
- Achieved an accuracy score of approximately 85.91% on the test data.
- Identified key words and phrases associated with different sentiment classes through word cloud visualization.
- Conducted hyperparameter tuning to optimize the SVM classifier's performance.

#### Usage:
- Clone the repository.
- Install the required libraries listed in the `requirements.txt` file.
- Run the Jupyter Notebook or Python script to replicate the sentiment analysis process.
- Experiment with different classifiers or hyperparameters to improve model performance.

#### References:
- Provide links to relevant articles, papers, or datasets used in the project.
#### DataSet -- 
  Link : https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets/data






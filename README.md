
# News Sentiment Analysis

The onset of the digital age has made the global community more interconnected than ever. With increased accessibility to information, it has become even easier to track events occurring in any part of the world as they unfold. Accordingly, the sources of information with the largest platforms and the fastest publishing time are those which maintain the greatest dominance over the development of global narratives. The nature of the news cycle, which rewards timeliness over accuracy, makes the current state of the media industry problematic, as it has a greater tendency to spread misinformation and perpetuate biases that reinforce a particular worldview. This project seeks to explore how this phenomenon varies amongst prominent news outlets in different countries as global events transpire, with a specific focus on articles published at the start of the Israel–Palestine Conflict in October 2023. The work applies sentiment analysis to evaluate headlines and identify potential biases in their phrasing. Additionally, classification models are built in order to test if it is possible to make predictions based on various features of the data set. The overall goal is to determine whether it is possible to identify particular rhetorical inclinations of a given news outlet, as this could provide further insight into the role of the media in shaping perceptions of global events.


## Data Download and Ingest

To access the project software, download the Smaliak_Final_Software.ipynb as a Colab notebook in Google Drive.

In order to upload the dataset, the GDELT Database file in this project must be downloaded to the local device as an excel sheet. Once the dataset is downloaded as an excel on the local device, it must be uploaded to the files section on the left side of the Colab notebook. After this is completed, the notebook will be able to load the dataset into a dataframe.

## Pre and Post Processing 

Before running the command which transforms the dataset into the dataframe, it is important to process all of the necessary imports that will allow the code to compile. 

    # Imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.svm import SVC

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    from bs4 import BeautifulSoup

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')

    import requests
    import re

    from wordcloud import WordCloud
    from gensim.models import Word2Vec

Since the dataset was manually extracted with its desired features, there are no extra steps needed to process it, outside of the data cleaning, feature engineering, and data transformation done in the Colab.  

The rest of the notebook should compile accordingly when run.

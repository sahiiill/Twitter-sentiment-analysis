import tweepy
import re
import pandas as pd
import configparser
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st
import datetime, pytz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from textblob import TextBlob
import joblib

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def twitter_connection():

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config.ini")
        
    api_key = ''
    api_key_secret = ''
    access_token = ''
    access_token_secret = ''

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    api = tweepy.API(auth)

    return api

api = twitter_connection()

def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub("\n","",text) # Removing hyperlink
    text = re.sub(":","",text) # Removing hyperlink
    text = re.sub("_","",text) # Removing hyperlink
    text = emoji_pattern.sub(r'', text)
    return text

def extract_mentions(text):
    text = re.findall("(@[A-Za-z0–9\d\w]+)", text)
    return text

def extract_hastag(text):
    text = re.findall("(#[A-Za-z0–9\d\w]+)", text)
    return text

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity
 
def train_svm_model(data):
    # Vectorize the text data using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Tweets'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, data['Analysis'], test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    accuracy = svm.score(X_test, y_test)
    # Save the trained SVM model to a file
    joblib.dump(svm_model, 'svm_model.pkl')

    return svm_model, accuracy

# def getAnalysis(score):
#   if score < 0:
#     return 'Negative'
#   elif score == 0:
#     return 'Neutral'
#   else:
#     return 'Positive'


def getAnalysisTextBlob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'
      
def getAnalysisSVM(text, svm_model, vectorizer):
    X = vectorizer.transform([text])
    y_pred = svm_model.predict(X)
    if y_pred == 0:
        return 'Negative'
    elif y_pred == 1:
        return 'Neutral'
    else:
        return 'Positive'
  


@st.cache_data()
def preprocessing_data(word_query, number_of_tweets, function_option, algorithm):

  if function_option == "Search By #Tag and Words":
    posts = tweepy.Cursor(api.search_tweets, q=word_query, count = 200, lang ="en", tweet_mode="extended").items((number_of_tweets))
  
  if function_option == "Search By Username":
    posts = tweepy.Cursor(api.user_timeline, screen_name=word_query, count = 200, tweet_mode="extended").items((number_of_tweets))
  
  data  = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

  if algorithm == "TextBlob":
      data['Analysis'] = data['Tweets'].apply(getAnalysisTextBlob)
  elif algorithm == "SVM":
      svm_model = joblib.load('svm_model.pkl')
      vectorizer = joblib.load('vectorizer.pkl')
      data['Tweets'] = data['Tweets'].apply(cleanTxt)
      data['Analysis'] = data['Tweets'].apply(getAnalysisSVM, svm_model=svm_model, vectorizer=vectorizer)
      # X_test = vectorizer.transform(data['Tweets'])
      # y_pred = svm_model.predict(X_test)
      # data['Analysis'] = y_pred
  
  data["mentions"] = data["Tweets"].apply(extract_mentions)
  data["hastags"] = data["Tweets"].apply(extract_hastag)
  data['links'] = data['Tweets'].str.extract('(https?:\/\/\S+)', expand=False).str.strip()
  data['retweets'] = data['Tweets'].str.extract('(RT[\s@[A-Za-z0–9\d\w]+)', expand=False).str.strip()

  data['Tweets'] = data['Tweets'].apply(cleanTxt)
  discard = ["CNFTGiveaway", "GIVEAWAYPrizes", "Giveaway", "Airdrop", "GIVEAWAY", "makemoneyonline", "affiliatemarketing"]
  data = data[~data["Tweets"].str.contains('|'.join(discard))]

  data['Subjectivity'] = data['Tweets'].apply(getSubjectivity)
  data['Polarity'] = data['Tweets'].apply(getPolarity)

  
  
  
  # # Split the data into training and testing sets
  # X_train, X_test, y_train, y_test = train_test_split(data['Tweets'], data['Analysis'], test_size=0.2, random_state=42)

  # # Convert the tweets into a matrix of TF-IDF features
  # vectorizer = TfidfVectorizer()
  # X_train = vectorizer.fit_transform(X_train)
  # X_test = vectorizer.transform(X_test)

  # # Train the SVM model
  # svm_model = svm.SVC(kernel='linear')
  # svm_model.fit(X_train, y_train)

  # # Evaluate the SVM model using the testing data
  # y_pred = svm_model.predict(X_test)
  # print(classification_report(y_test, y_pred))
  

  return data


def download_data(data, label):
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = "{}.{}-{}-{}".format(current_time.date(), current_time.hour, current_time.minute, current_time.second)
    export_data = st.download_button(
                        label="Download {} data as CSV".format(label),
                        data=data.to_csv(),
                        file_name='{}{}.csv'.format(label, current_time),
                        mime='text/csv',
                        help = "When You Click On Download Button You can download your {} CSV File".format(label)
                    )
    return export_data


def analyse_mention(data):

  # mention = pd.DataFrame(data["mentions"].to_list()).add_prefix("mention_")
  
  # try:
  mention = pd.DataFrame(data["mentions"].to_list()).add_prefix("mention_")
  # except KeyError:
    # return pd.Series(dtype=int)

  try:
    mention = pd.concat([mention["mention_0"], mention["mention_1"], mention["mention_2"]], ignore_index=True)
  except:
    mention = pd.concat([mention["mention_0"]], ignore_index=True)
    # return pd.Series(dtype=int)
  
  mention = mention.value_counts().head(10)
  
  return mention



def analyse_hastag(data):
  
  hastag = pd.DataFrame(data["hastags"].to_list()).add_prefix("hastag_")

  try:
    hastag = pd.concat([hastag["hastag_0"], hastag["hastag_1"], hastag["hastag_2"]], ignore_index=True)
  except:
    hastag = pd.concat([hastag["hastag_0"]], ignore_index=True)
  
  hastag = hastag.value_counts().head(10)

  return hastag




def graph_sentiment(data):

  analys = data["Analysis"].value_counts().reset_index().sort_values(by="index", ascending=False)
  
  return analys
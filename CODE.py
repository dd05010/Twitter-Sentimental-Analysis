# ### Import Libraries and Modules

import tweepy as tw
from tweepy import Cursor
from textblob import TextBlob
import pandas as pd 
import numpy as np
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



nltk.download('punkt')
nltk.download('wordnet')


# ### Twitter Authenticater

twitter_consumer_key = 'XX'
twitter_consumer_secret = 'XX'
twitter_access_token = 'XX'
twitter_access_secret = 'XX'


auth = tw.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# ### Stream and Process live data

search_name = 'CovidVaccine'
new_name = search_name + " -filter:retweets"

tweets = tw.Cursor(api.search,
              q=new_name,
              lang="en").items(100)

    
df = pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])

df.head()



# ### Remove Noise and Clean Data

def cleanTweets(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    txt = re.sub(r'#', '', txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    return txt


stop_words = set(stopwords.words('english'))
def preprocessText(txt):
    txt = txt.translate(str.maketrans('','', string.punctuation))
    txt_tokens = word_tokenize(txt)
    filtered = [word for word in txt_tokens if not word in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(word, pos='a') for word in stemmed_words]
    return " ".join(filtered)


df['Tweet']=df['Tweet'].apply(cleanTweets)
df['Tweet']=df['Tweet'].apply(preprocessText)


def getTweetSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTweetPolarity(txt):
    return TextBlob(txt).sentiment.polarity


df['Subjectivity']=df['Tweet'].apply(getTweetSubjectivity)
df['Polarity']=df['Tweet'].apply(getTweetPolarity)
df.head(50)

df = df.drop_duplicates()
df = df.drop(df[df['Tweet'] == ''].index)


# ### Analyze Data

def getTweetAnalysis(polarity):
    if polarity>0:
        return "Positive"
    elif polarity==0:
        return "Neutral"
    else:
        return "Negative"
    

df["Sentiment"]=df["Polarity"].apply(getTweetAnalysis)
df.head(50)


positiv=df[df['Sentiment']=="Positive"]
pst=((positiv.shape[0])/(df.shape[0]))*100
print(pst)


negativ=df[df['Sentiment']=="Negative"]
ngt=((negativ.shape[0])/(df.shape[0]))*100
print(ngt)


neutral=df[df['Sentiment']=="Neutral"]
ntr=((neutral.shape[0])/(df.shape[0]))*100
print(ntr)


# ### Creating WordCloud

words = ' '.join([tweets for tweets in df['Tweet']])
def showWordcloud(words):
    wordCloud = WordCloud(
        width=800,
        height=500,
        max_words=200,
        max_font_size=80).generate(words)
        
    
plt.figure(figsize=(12,10)) 
plt.imshow(wordCloud)
plt.axis('off')
plt.show()


# ### Pie Chart

values = ['Positive','Negative','Neutral']
percentage = [pst,ngt,ntr]
explode = (0.1, 0, 0)
color=['green','red','yellow']

plt.pie(percentage,colors=color, labels=values,explode=explode, autopct='%0.f%%', shadow=True, startangle=600)
plt.axis('equal')
plt.legend(values,loc='lower right',title='Sentiments',bbox_to_anchor =(1, 0, 0.5, 1))


# ### Bar Chart


labels = df.groupby('Sentiment').count().index.values
values = df.groupby('Sentiment').size().values
color=['red','yellow','green']
plt.bar(labels,values,color=color)






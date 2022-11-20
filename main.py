from textblob import TextBlob
from dotenv import load_dotenv
import os
import tweepy
import cld3
from translate import Translator
import numpy as np
import re

load_dotenv()

bearer_token = os.getenv('BEARER_TOKEN')

client = tweepy.Client(bearer_token)


def get_tweets(us_query: str):  # pull tweets

    us_query = f'{us_query} -is:retweet'
    response = client.search_recent_tweets(query=us_query, max_results=100, sort_order="recency")
    tweets = response.data
    return (tweets)


def drop_links(text):  # drop links
    return re.sub('http://\S+|https://\S+', '', text)


def drop_ids(text):  # drop links
    return re.sub('@\S+', '', text)


def clean_tweet(text):
    text = drop_links(text)
    text = drop_ids(text)
    return text


def is_english(text):  # check language
    if cld3.get_language(text).language == 'en':
        return True
    return False


def tweet_analysis(user_query: str):
    tweets = get_tweets(user_query)
    polarities = []
    subjectivities = []
    for tweet in tweets:
        phrase = TextBlob(tweet.text)

        tweet.text = clean_tweet(tweet.text)
        if tweet.text == '':
            continue

        if not is_english(tweet.text):
            #         print("Start Translation",tweet.text,sep = "\n")
            result = False
            while not result:
                try:
                    # connect to translator api
                    translator = Translator(to_lang="English")
                    translation = translator.translate(tweet.text)
                    result = True
                except:
                    pass

            phrase = TextBlob(translation)

        if phrase.sentiment.polarity != 0.0 and phrase.sentiment.subjectivity != 0.0:
            polarities.append(phrase.sentiment.polarity)
            subjectivities.append(phrase.sentiment.subjectivity)
        print('Tweet: ' + tweet.text)
        print("Polarity: " + str(phrase.sentiment.polarity) + ' \ Subjectivity: ' + str(phrase.sentiment.subjectivity))
        print('.....................')

    print("Mean polarity: ", np.mean(polarities), "Count: ", len(polarities))
    print("Std polarity: ", np.std(polarities))

    return {'polarity': polarities, 'subjectivity': subjectivities}


def get_weighted_polarity_mean(valid_tweets):
    return np.average(valid_tweets['polarity'], weights=valid_tweets['subjectivity'])


def print_result(mean):
    if mean > 0.0:
        print('POSITIVE')
    elif mean == 0.0:
        print('NEUTRO')
    else:
        print('NEGATIVE')


if __name__ == "__main__":
    query = input("Query: ")
    print("Analysis started...")
    analysis = tweet_analysis(query)

    print('WEIGHTED MEAN: ' + str(get_weighted_polarity_mean(analysis)))
    print_result(get_weighted_polarity_mean(analysis))
    #
    # print('MEAN: ' + str(get_polarity_mean(analysis)))
    # print_result(get_polarity_mean(analysis))

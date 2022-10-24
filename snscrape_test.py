import snscrape.modules.twitter as sntwitter
import pandas as pd

# Creating list to append tweet data to
tweets_list = []
accounts_list = ["CNBC", "FT", "Reuters"]
keywords_list = ["Tesla", "Apple", "Amazon"]

# Using TwitterSearchScraper to scrape data and append tweets to list
# Currently adds all tweets from CNBC containing the keyword 'Tesla' that are not retweets or replies
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Tesla since:2010-01-01 until:2022-12-31 from:CNBC').get_items()):
    if i>20000:
        break
    if "@" not in tweet.content:
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
tweets_df.to_csv('CNBC_tesla_tweets.csv')
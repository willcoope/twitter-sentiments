import snscrape.modules.twitter as sntwitter
import pandas as pd

# Create list of accounts to search
accounts_list = ["CNBC", "FT", "Reuters"]

# Iterate across accounts listed
for account in accounts_list:
    # Create empty list to store scraped tweets
    tweets_list = []
    # Search for tweets since 2010 from the specified account whose text contains the keyword 'Tesla'
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Tesla since:2010-01-01 until:2022-12-31 from:' + account).get_items()):
        if i>20000:
            break
        # Disallow replies and retweets
        if "@" and "RT" not in tweet.content:
            # Add relevant data to list
            tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
        # Create a dataframe from the tweets_list above
        tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
        # Save dataframe in a CSV file with custom name according to account scraped
        tweets_df.to_csv(account + '_tesla_tweets.csv')
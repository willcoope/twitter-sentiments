import snscrape.modules.twitter as sntwitter
import pandas as pd

# Create list of accounts to search
accounts_list = ["CNBC","FT", "Reuters"]
stocks_list = ["Tesla", "Amazon"]

#Iterate across accounts listed
for account in accounts_list:
    for stock in stocks_list:
    # Create empty list to store scraped tweets
        tweets_list = []
        # Search tweets since 2013 from the specified account whose text contains the keyword
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(stock + 'since:2013-01-01 until:2023-12-31 from:' + account).get_items()):
            if i>20000:
                break
            # Disallow replies and retweets
            if "@" and "RT" not in tweet.content:
                # Add relevant data to list
                tweets_list.append([tweet.user.username, tweet.date, tweet.content])
            # Create dataframe from the tweets_list above
            tweets_df = pd.DataFrame(tweets_list, columns=['Username', 'Datetime', 'Text'])
            # Convert dataframe to CSV file with custom name
            tweets_df.to_csv(account + '_amazon_tweets.csv')
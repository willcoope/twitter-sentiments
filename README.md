# twitter-sentiments
Twitter Sentiments consists of a single Python file and a collection of CSV files, each containing the stock prices of a company each week from the 1st January 2013 until the 1st January 2023. These files are used by the sentiment_analysis.py file in order to graph the correlation between sentiment online towards a company and the stock price that week.

Requirements:
Python - version 3 or later

Steps:
1. Clone the repository
2. Run the following command:
    python -c 'import sentiment_analysis; sentiment_analysis.twitter_scrape()'
    
This will create a CSV file for each of the stocks, containing every tweet referencing that company from the financial news Twitter accounts listed.
3. Run the following command:
    python3 sentiment_analysis.py

This will use the all-data.csv file to train the Naive Bayes classifier to analyse the sentiments of tweets, then analyse all the tweets for each company and generate graphs showing stock prices each week and corresponding sentiments.

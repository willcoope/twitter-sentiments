from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import re, string, random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Remove hyperlinks, punctuation and special characters from tokens
# Convert remaining tokens to a normal form (losing becomes lose, profits becomes profit)

def remove_noise(news_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(news_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_headlines_for_model(cleaned_tokens_list):
    for news_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in news_tokens)

# Scrape data from csv file and separate by sentiment

positive_headline_tokens = []
negative_headline_tokens = []
df = pd.read_csv('all-data.csv', encoding = "ISO-8859-1")
df.reset_index()
for index, row in df.iterrows():
    if (row[0] == "positive"):
        positive_headline_tokens.append(row[1].split())
    if (row[0] == "negative"):
        negative_headline_tokens.append(row[1].split())

df = pd.read_csv('all-data2.csv', encoding = "ISO-8859-1")
df.reset_index()
for index, row in df.iterrows():
    if (row[1] == "positive"):
        positive_headline_tokens.append(row[0].split())
    if (row[1] == "negative"):
        negative_headline_tokens.append(row[0].split())

# Clean tokens and set sentiment for each headline

own_positive_cleaned_tokens_list = []
own_negative_cleaned_tokens_list = []

stop_words = stopwords.words('english')

for tokens in positive_headline_tokens:
    own_positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_headline_tokens:
    own_negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

positive_dataset = [(headline, "Positive")
                         for headline in get_headlines_for_model(own_positive_cleaned_tokens_list)]
negative_dataset = [(headline, "Negative")
                     for headline in get_headlines_for_model(own_negative_cleaned_tokens_list)]

# Function that checks the month of all data in a csv
# If the month is correct and headline is classified as positive,
# Increment positive count by 1.

# Combine data and randomly split
# Create model with training data then test

dataset = positive_dataset + negative_dataset
print("Dataset Length:", len(dataset))
random.shuffle(dataset)
train_data = dataset[:4100]
test_data = dataset[4100:]
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(25))

# Separate tweets into months and creates pairs of values
# Each month has a corresponding percentage of tweets that are positive

def headline_analysis(csv_file, total_headlines_predicted, weekly_headlines_count, weekly_positive_headlines_count, weekly_negative_headlines_count):
    with open(csv_file) as file_obj:
    # Create reader object by passing the file object to reader method
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                if(row[2] > current_week_str and row [2] < next_week_str):
                    weekly_headlines_count += 1
                    total_headlines_predicted += 1
                    custom_tokens = remove_noise(word_tokenize(row[3]))
                    if (classifier.classify(dict([token, True] for token in custom_tokens))== "Positive"):
                        weekly_positive_headlines_count += 1
                    if (classifier.classify(dict([token, True] for token in custom_tokens))== "Negative"):
                        weekly_negative_headlines_count += 1
    return (total_headlines_predicted, weekly_headlines_count, weekly_positive_headlines_count, weekly_negative_headlines_count)

def data_plot(total_headlines_predicted, analysis_output, weekly_percentage_sentiment_price_pairs, weekly_net_sentiment_price_pairs):
    total_headlines_predicted = analysis_output [0]
    if (analysis_output[1] == 0):
        weekly_percentage_sentiment_price_pairs.update({current_week_str:0})
        weekly_net_sentiment_price_pairs.update({current_week_str:0})
    else:
        positive_percentage = (analysis_output[2] / analysis_output[1])*100
        net_sentiment = analysis_output[2] - analysis_output[3]
        weekly_percentage_sentiment_price_pairs.update({current_week_str:positive_percentage})
        weekly_net_sentiment_price_pairs.update({current_week_str:net_sentiment})
    return total_headlines_predicted

tesla_weekly_percentage_sentiment_price_pairs = {}
tesla_weekly_net_sentiment_price_pairs = {}

amazon_weekly_percentage_sentiment_price_pairs = {}
amazon_weekly_net_sentiment_price_pairs = {}

current_week = datetime(2013,1,1)
current_week_str = current_week.strftime("%Y-%m-%d")
total_headlines_predicted = 0
while(current_week_str < "2022-10-01"):
    next_week = current_week + relativedelta(weeks=1)
    current_week_str = current_week.strftime("%Y-%m-%d")
    next_week_str = next_week.strftime("%Y-%m-%d")
    analysis_output = headline_analysis('Tesla_all_tweets.csv', total_headlines_predicted, 0, 0, 0)
    total_headlines_predicted = data_plot(total_headlines_predicted, analysis_output, tesla_weekly_percentage_sentiment_price_pairs, tesla_weekly_net_sentiment_price_pairs)
    analysis_output = headline_analysis('Amazon_all_tweets.csv', total_headlines_predicted, 0, 0, 0)
    total_headlines_predicted = data_plot(total_headlines_predicted, analysis_output, amazon_weekly_percentage_sentiment_price_pairs, amazon_weekly_net_sentiment_price_pairs)
    current_week = current_week + relativedelta(weeks=1)
print("Total Headlines Predicted:", total_headlines_predicted)

# Tesla Weekly Graphs
########################################################################

x1 = []
y1 = []
with open('TSLA_weekly.csv') as file_obj:
    next(file_obj)
    # Create reader object by passing the file object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        if (row[0] >= "2013-01-01"):
            x1.append(row[0])
            y1.append(float(row[4]))
x2 = []
y2 = []
for k, v in tesla_weekly_percentage_sentiment_price_pairs.items():
    x2.append(k)
    y2.append(v)

# Create figure and axis objects with subplots()

fig,ax=plt.subplots()

# Make a plot for the stock price line chart
ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("TSLA Stock Price (USD)", color = "r", fontsize = 14)

# Make a plot with different y-axis using second axis object for the sentiment bar chart

ax2=ax.twinx()
ax2.bar(x2, y2, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Tesla Positive Sentiment Each Week (%)",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 2)
plt.show()

x3 = []
y3 = []
for k, v in tesla_weekly_net_sentiment_price_pairs.items():
    x3.append(k)
    y3.append(v)

fig,ax=plt.subplots()
ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("TSLA Stock Price (USD)", color = "r", fontsize = 14)

ax2=ax.twinx()
ax2.bar(x3, y3, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Tesla Net Sentiment Per Week",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 3)
plt.show()

# Amazon Weekly Graphs
########################################################################

x1 = []
y1 = []
with open('AMZN_weekly.csv') as file_obj:
    next(file_obj)
    # Create reader object by passing the file object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        if (row[0] >= "2013-01-01"):
            x1.append(row[0])
            y1.append(float(row[4]))
x2 = []
y2 = []
for k, v in amazon_weekly_percentage_sentiment_price_pairs.items():
    x2.append(k)
    y2.append(v)

# Create figure and axis objects with subplots()

fig,ax=plt.subplots()

# Make a plot for the stock price line chart
ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("AMZN Stock Price (USD)", color = "r", fontsize = 14)

# Make a plot with different y-axis using second axis object for the sentiment bar chart

ax2=ax.twinx()
ax2.bar(x2, y2, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Amazon Positive Sentiment Each Week (%)",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 2)
plt.show()

x3 = []
y3 = []
for k, v in amazon_weekly_net_sentiment_price_pairs.items():
    x3.append(k)
    y3.append(v)

fig,ax=plt.subplots()
ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("AMZN Stock Price (USD)", color = "r", fontsize = 14)

ax2=ax.twinx()
ax2.bar(x3, y3, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Amazon Net Sentiment Per Week",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 3)
plt.show()
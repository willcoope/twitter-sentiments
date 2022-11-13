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

def headline_analysis(csv_file, total_headlines_predicted, monthly_headlines_count, monthly_positive_headlines_count, monthly_negative_headlines_count):
    with open(csv_file) as file_obj:
    # Create reader object by passing the file object to reader method
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                if(row[2] > current_month_str and row [2] < next_month_str):
                    monthly_headlines_count += 1
                    total_headlines_predicted += 1
                    custom_tokens = remove_noise(word_tokenize(row[3]))
                    if (classifier.classify(dict([token, True] for token in custom_tokens))== "Positive"):
                        monthly_positive_headlines_count += 1
                    if (classifier.classify(dict([token, True] for token in custom_tokens))== "Negative"):
                        monthly_negative_headlines_count += 1
    return (total_headlines_predicted, monthly_headlines_count, monthly_positive_headlines_count, monthly_negative_headlines_count)

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

percentage_sentiment_price_pairs = {}
net_sentiment_price_pairs = {}

current_month = datetime(2013,1,1)
current_month_str = current_month.strftime("%Y-%m-%d")
total_headlines_predicted = 0
while(current_month_str < "2022-10-01"):
    next_month = current_month + relativedelta(months=1)
    current_month_str = current_month.strftime("%Y-%m-%d")
    next_month_str = next_month.strftime("%Y-%m-%d")
    monthly_headlines_count = 0
    monthly_positive_headlines_count = 0
    monthly_negative_headlines_count = 0
    output = headline_analysis('CNBC_tesla_tweets.csv', total_headlines_predicted, 0, 0, 0)
    #print("Monthly Headlines:", output[1])
    #print("Total Headlines:", output[0])
    output = headline_analysis('FT_tesla_tweets.csv', output [0], output [1], output [2], output[3])
    #print("Monthly Headlines:", output[1])
    #print("Total Headlines:", output[0])
    output = headline_analysis('Reuters_tesla_tweets.csv', output [0], output [1], output [2], output[3])
    #print("Monthly Headlines:", output[1])
    #print("Total Headlines:", output[0])
    total_headlines_predicted = output [0]
    if (output[1] == 0):
        percentage_sentiment_price_pairs.update({current_month_str:0})
        percentage_sentiment_price_pairs.update({current_month_str:0})
    else:
        positive_percentage = (output[2] / output[1])*100
        net_sentiment = output[2] - output[3]
        #print(current_month_str, "monthly_positive_headlines_count", output[2])
        #print(current_month_str, "monthly_negative_headlines_count", output[3])
        #print(current_month_str, "Net Sentiment:", net_sentiment)
        percentage_sentiment_price_pairs.update({current_month_str:positive_percentage})
        net_sentiment_price_pairs.update({current_month_str:net_sentiment})
    current_month = current_month + relativedelta(months=1)
#print("Headlines Predicted: ", total_headlines_predicted)
#print("net_sentiment_price_pairs")
#(net_sentiment_price_pairs)
 
# Create lists of X and Y values for the TSLA stock prices and dates
 
x1 = []
y1 = []
with open('TSLA_monthly.csv') as file_obj:
    next(file_obj)
    # Create reader object by passing the file object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        if (row[0] >= "2013-01-01"):
            x1.append(row[0])
            y1.append(float(row[4]))

# Create lists of X and Y values for positive sentiment percentages and dates

x2 = []
y2 = []
for k, v in percentage_sentiment_price_pairs.items():
    x2.append(k)
    y2.append(v)

# Create figure and axis objects with subplots()

fig,ax=plt.subplots()

# Make a plot for the stock price line chart

ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price (USD)", color = "r", fontsize = 14)

# Make a plot with different y-axis using second axis object for the sentiment bar chart

ax2=ax.twinx()
ax2.bar(x2, y2, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Positive Sentiment Each Month (%)",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 3)
plt.show()

# Positive and Negative Bar Chart Example

# x = range(7)
# negative_data = [-1,-4,-3,-2,-6,-2,-8]
# positive_data = [4,2,3,1,4,6,7,]

# fig = plt.figure()
# ax = plt.subplot(111)
# ax.bar(x, negative_data, width=1, color='r')
# ax.bar(x, positive_data, width=1, color='b')
# plt.show()

x3 = []
y3 = []
for k, v in net_sentiment_price_pairs.items():
    x3.append(k)
    y3.append(v)

fig,ax=plt.subplots()

# Make a plot for the stock price line chart

ax.plot(x1, y1, color = 'r', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price (USD)", color = "r", fontsize = 14)
ax2=ax.twinx()
ax2.bar(x3, y3, color = 'b', label = "Sentiment", alpha = 0.5)
ax2.set_ylabel("Net Sentiment Per Month",color="b",fontsize=14)
ax.tick_params(axis = "x", rotation = 90, labelsize = 3)
plt.show()
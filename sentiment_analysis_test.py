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

# Combine data and randomly split
# Create model with training data then test

dataset = positive_dataset + negative_dataset
print("Dataset Length:", len(dataset))
random.shuffle(dataset)
train_data = dataset[:1800]
test_data = dataset[1800:]
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(25))

# Open CSV file
sentiment_price_pairs = {}
# with open('CNBC_tesla_tweets.csv') as file_obj:

#     # Create reader object by passing the file object to reader method

#     reader_obj = csv.reader(file_obj)

    # Iterate over each row in the csv file using reader object, removing noise and classifying
    #current_month = datetime(2021, 1, 1)
    #next_month = current_month + relativedelta(months=1)
    # current_month = datetime.datetime(2021, 1, 1, 0, 0)
    # next_month = current_month + relativedelta(months=1)
    # current_month.strftime('%d/%m/%Y')
    # next_month.strftime('%d/%m/%Y')
    # print(type(current_month))
    # current date and time
# current_month = datetime(2021,2,1) -- this works pretty well
current_month = datetime(2013,1,1)
current_month_str = current_month.strftime("%Y-%m-%d")
while(current_month_str < "2022-10-01"):
    next_month = current_month + relativedelta(months=1)
    #print(type(current_month))
    #print(type(next_month))
    current_month_str = current_month.strftime("%Y-%m-%d")
    #print("current_month_str:", current_month_str)
    #print(type(current_month_str))
    next_month_str = next_month.strftime("%Y-%m-%d")
    #print("next_month_str:", next_month_str)
    #print(type(next_month_str))
    total_headlines_count = 0
    positive_headlines_count = 0
    with open('CNBC_tesla_tweets.csv') as file_obj:

    # Create reader object by passing the file object to reader method

            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                if(row[2] > current_month_str and row [2] < next_month_str):
                    total_headlines_count += 1
                    #print(type(row[2]))
                    custom_tokens = remove_noise(word_tokenize(row[3]))
                    if (classifier.classify(dict([token, True] for token in custom_tokens))== "Positive"):
                        positive_headlines_count += 1
            #print("Positive headlines: ", positive_headlines_count)
            #print("Total headlines: ", total_headlines_count)
            if (total_headlines_count == 0 or positive_headlines_count == 0):
                sentiment_price_pairs.update({current_month_str:0})
            else:
                positive_percentage = (positive_headlines_count / total_headlines_count)*100
                #print("Positive percentage for ", current_month_str ,": ", positive_percentage)
                sentiment_price_pairs.update({current_month_str:positive_percentage})
            current_month = current_month + relativedelta(months=1)
#print(sentiment_price_pairs)



 
x1 = []
y1 = []
  
with open('TSLA_monthly.csv') as file_obj:
    next(file_obj)
    # Create reader object by passing the file object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        if (row[0] >= "2015-01-01"):
        #print((row[0]))
            x1.append(row[0])
            y1.append(float(row[4]))
print(x1)
print(y1)
  
x2 = []
y2 = []
for k, v in sentiment_price_pairs.items():
    x2.append(k)
    y2.append(v)
print(x2)
print(y2)
# create figure and axis objects with subplots()
fig,ax=plt.subplots()
ax.plot(x1, y1, color = 'g', label = "Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

# # twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# # make a plot with different y-axis using second axis object
ax2.bar(x2, y2, color = 'b', label = "Sentiment")
ax2.set_ylabel("Sentiment Percentage",color="b",fontsize=14)
# #plt.show()
# # save the plot as a file
# # fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
# #             format='jpeg',
# #             dpi=100,
# #             bbox_inches='tight')
# #ticks = []
ax.tick_params(axis = "x", rotation = 90, labelsize = 2)
plt.show()

# plt.plot(x1, y1, color = 'g', label = "Price")
# plt.bar(x2, y2, color = 'r', label = "Sentiment")
# ticks = []
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.xlabel('Date')
# plt.tick_params(labelsize=4)
# plt.ylabel('Price')
# plt.title('Stock Price')
# plt.legend()
# plt.show()
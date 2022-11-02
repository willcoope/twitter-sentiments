from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import classify, NaiveBayesClassifier
import re, string, random
import pandas as pd

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

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_headlines_for_model(cleaned_tokens_list):
    for news_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in news_tokens)

own_positive_tokens = []
own_negative_tokens = []
df = pd.read_csv('all-data.csv', encoding = "ISO-8859-1")
df.reset_index()
for index, row in df.iterrows():
    if (row[0] == "positive"):
        own_positive_tokens.append(row[1].split())
    if (row[0] == "negative"):
        own_negative_tokens.append(row[1].split())

own_positive_cleaned_tokens_list = []
own_negative_cleaned_tokens_list = []
stop_words = stopwords.words('english')

for tokens in own_positive_tokens:
    own_positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in own_negative_tokens:
    own_negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

positive_dataset = [(headline, "Positive")
                         for headline in get_headlines_for_model(own_positive_cleaned_tokens_list)]
negative_dataset = [(headline, "Negative")
                     for headline in get_headlines_for_model(own_negative_cleaned_tokens_list)]
dataset = positive_dataset + negative_dataset

print("Dataset Length:", len(dataset))
random.shuffle(dataset)
train_data = dataset[:1800]
test_data = dataset[1800:]
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(20))
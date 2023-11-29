import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
from nltk import tokenize, stem, sentiment
from nltk.corpus import stopwords, wordnet
import nltk as nltk

def main():
    df = pd.read_csv("Spam_Email_Classification\combined_data.csv")
    labels = list(df["label"])
    txt_preproc = PreProcessingEmail(df['text'])
    cleaned_text = \
        txt_preproc \
        .clean_text() \
        .get_processed_text()
    processed_txt = \
        cleaned_text \
        .tokenize_text() \
        .remove_stopwords() \
        .lemmatize_words() \
        .get_processed_text()
    NER = \
        cleaned_text \
        .perform_NER() \
        .get_NER() 
    sentiment = \
        cleaned_text \
        .extract_sentiment() \
        .get_sentiment()    
    np.savetxt('data.csv', (processed_txt, NER, sentiment), delimiter=',')

    
def download_if_non_existent(res_path, res_name):
  try:
    nltk.data.find(res_path)
  except LookupError:
    print(f'resource {res_path} not found. Downloading now...')
    nltk.download(res_name)

class PreProcessingEmail:
    def __init__(self, X):
        self.X = X
        self.sentiment = []
        self.NER = []
        download_if_non_existent('corpora/stopwords', 'stopwords')
        download_if_non_existent('tokenizers/punkt', 'punkt')
        download_if_non_existent('taggers/averaged_perceptron_tagger',
                                'averaged_perceptron_tagger')
        download_if_non_existent('corpora/wordnet', 'wordnet')
        download_if_non_existent('corpora/omw-1.4', 'omw-1.4')

        self.sw_nltk = stopwords.words('english')
        new_stopwords = ['<*>']
        self.sw_nltk.extend(new_stopwords)
        self.sw_nltk.remove('not')
    
    # function to clean text
    def clean_text(self):
        # remove non-alphanumeric characters
        self.X = self.X.apply(lambda x: re.sub(r'\W+', '', x))
        # remove digits
        self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
        # remove extra spaces
        self.X = self.X.apply(lambda x: re.sub(r'\s+', '', x))
        # remove escape numbers and escape chars
        self.X = self.X.apply(lambda x: re.sub(r'(escapenumber)+', '', x))
        self.X = self.X.apply(lambda x: re.sub(r'(escapelong)+', '', x))
        # convert to lowercase
        self.X = self.X.apply(lambda x: str.lower(x))
        return self

    # function to tokenize text
    def tokenize_text(self):
        # tokenize sentences
        sentences = tokenize.sent_tokenize(self.X)
        # tokenize words
        words = []
        for sentence in sentences:
            words += tokenize.word_tokenize(sentence)
        self.X = words
        return self

    # function to remove stopwords
    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        self.X = words
        return self

    # function to lemmatize words
    def lemmatize_words(self):
        lemmatizer = stem.WordNetLemmatizer()
        lemmatized_words = []
        for word, tag in nltk.pos_tag(self.X):
            # map POS tag to first letter used by WordNetLemmatizer
            tag = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else wordnet.NOUN
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))
        self.X = lemmatized_words
        return self

    # function to perform NER
    def perform_ner(self):
        self.NER = nltk.ne_chunk(nltk.pos_tag(tokenize.word_tokenize(self.X)))
        return self

    # function to extract sentiment
    def extract_sentiment(self):
        sia = sentiment.SentimentIntensityAnalyzer()
        self.sentiment = sia.polarity_scores(self.X)
        return self
    
    def get_processed_text(self):
        return self.X
    
    def get_NER(self):
        return self.NER
    
    def get_sentiment(self):
        return self.sentiment
    
if __name__ == "__main__":
    main()
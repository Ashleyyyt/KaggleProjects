import preprocessing as pp
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("Spam_Email_Classification\combined_data.csv")
    labels = list(df["label"])
    txt_preproc = pp.PreProcessingEmail(df['text'])
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

if __name__ == "__main__":
    main()
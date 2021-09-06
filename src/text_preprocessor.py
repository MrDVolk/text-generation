import re
import string
import nltk
import spacy
from nltk.corpus import stopwords


class TextPreprocessor:
    def __init__(self, *, gentle=False, stopwords_removal=True, lemmatization=True, same_letter_collapsing=True):
        self.stopwords_removal = stopwords_removal
        self.lemmatization = lemmatization
        self.same_letter_collapsing = same_letter_collapsing
        
        if gentle:
            self.stopwords_removal = False
            self.lemmatization = False
            self.same_letter_collapsing = False
        
        if self.stopwords_removal:        
            nltk.download('stopwords')
            english_stopwords = stopwords.words('english')
            self.stopwords = set(english_stopwords)
            
        if self.lemmatization:
            self.nlp = spacy.load('en_core_web_sm')
        
    def collapse_same_letters(self, text):
        text = re.sub(r'([a-z])\1{2,}', '\g<1>', text)
        return text
    
    def remove_stop_words(self, text):
        words = text.split(' ')
        text = ' '.join([word for word in words if word not in self.stopwords])
        return text
    
    def lemmatize(self, text):
        text = ' '.join([w.lemma_ for w in self.nlp(text)])
        return text

    def process(self, text):
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        if self.same_letter_collapsing:
            text = self.collapse_same_letters(text)
            
        if self.stopwords_removal:
            text = self.remove_stop_words(text)

        text = re.sub(r'[^a-z ]', ' ', text)
        text = re.sub(r'[a-z]{35,}', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        
        if self.lemmatization:
            text = self.lemmatize(text)
            
        text = text.strip()
        return text

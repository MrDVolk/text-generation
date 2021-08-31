import re
import string
import nltk
import spacy
from nltk.corpus import stopwords


class TextPreprocessor:
    def __init__(self, *, gentle=False):
        self.gentle_processing = gentle
        if not gentle:        
            nltk.download('stopwords')
            english_stopwords = stopwords.words('english')
            self.stopwords = set(english_stopwords)
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

        if not self.gentle_processing:
            text = self.collapse_same_letters(text)    
            text = self.remove_stop_words(text)

        text = re.sub(r'[^a-z ]', ' ', text)
        text = re.sub(r'[a-z]{35,}', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        
        if not self.gentle_processing:
            text = self.lemmatize(text)
            
        text = text.strip()
        return text

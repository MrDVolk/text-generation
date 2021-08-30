import numpy as np
import mmap
from sklearn.neighbors import KDTree
from tqdm import tqdm


class EmbeddingManager:
    def __init__(self, path):
        def get_num_lines(file_path):
            fp = open(file_path, "r+")
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1
            return lines

        self.embeddings_dict = {}
        with open(path, 'r', encoding='utf-8') as file:
            skipped_header = False
            for line in tqdm(file, total=get_num_lines(path)):
                if not skipped_header:
                    skipped_header = True
                    continue
                    
                values = line.split()
                word = values[0].lower()
                # clean words - only letters (and numerals, just in case),
                # and every word that is longer then 35 letters is clearly a typo
                # https://en.wikipedia.org/wiki/Longest_word_in_English
                if not word.isalpha() and not word.isnumeric() or len(word) > 30:
                    continue
                
                vector = np.asarray(values[1:], dtype='float32')
                
                # average repeated word embeddings
                if word in self.embeddings_dict:
                    existing_vector = self.embeddings_dict[word]
                    average_vector = np.mean([vector, existing_vector], axis=0)
                    self.embeddings_dict[word] = average_vector
                else:
                    self.embeddings_dict[word] = vector
                    
        # because of we have to average repeaded word embeddings, we have to iterate one more time
        # to retrieve separate vectors and labels
        words = []
        vectors = []
        for key in self.embeddings_dict:
            words.append(key)
            vectors.append(self.embeddings_dict[key])
            
        self.words = np.array(words)
        self.vectors = np.array(vectors)
        self.shape = self.vectors[0].shape
        self.estimator = None
        print(f'Total embeddings shape: {self.vectors.shape}')
        
    def get_vector(self, word):
        word = word.lower()
        if word in self.embeddings_dict:
            return self.embeddings_dict[word]
        else:
            return np.zeros(self.shape)
        
    def get_words(self, vector, k=1):
        if vector.shape != self.shape:
            return [('', np.inf)]
        
        if self.estimator is None:
            print('Estimator is being prepared...')
            self.estimator = KDTree(self.vectors)
            
        distances, idx = self.estimator.query([vector], k=k)
        words = self.words[idx]
        zipped_result = list(zip(np.ravel(words), np.ravel(distances)))
        return zipped_result
    
    def add_special_vectors(self, special_vectors):
        for key in special_vectors:
            self.embeddings_dict[key] = special_vectors[key]
    
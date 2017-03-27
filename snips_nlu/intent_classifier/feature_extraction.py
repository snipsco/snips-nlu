from intent_classifier_resources import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2

class Featurizer(object):

    def __init__(self):

        self.count_vect = CountVectorizer(ngram_range=(1, 1))
        self.tfidf_transformer = TfidfTransformer()
        self.best_features = None

    def fit(self, queries, y):
        
        X_train_counts = self.count_vect.fit_transform(queries)
        list_index_words = {self.count_vect.vocabulary_[x]:x for x in self.count_vect.vocabulary_} 
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        stop_words = get_stop_words("en")
            
        chi2val, pval = chi2(X_train_tfidf, y)
        self.best_features = [i for i,v in enumerate(pval) if v < 0.4]
        if len(self.best_features) == 0:
            self.best_features = [idx for idx, val in enumerate(pval) if val == pval.min()] 

        feature_names = {}
        for i in self.best_features:
            feature_names[i] = { 'word': list_index_words[i] , 'pval': pval[i] }
            
        to_remove = []
        for feat in feature_names:
            if feature_names[feat]['word'] in stop_words:
                if feature_names[feat]['pval'] > 0.2:
                    self.best_features.remove(feat)
                    to_remove.append(feat)
        
        return self

    def transform(self, queries):
        X_train_counts = self.count_vect.transform(queries)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        X = X_train_tfidf[:, self.best_features]
        return X


    def fit_transform(self, queries, y):
        self = self.fit(queries, y)
        return self.transform(queries)

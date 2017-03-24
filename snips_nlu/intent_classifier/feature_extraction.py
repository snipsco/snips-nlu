from intent_classifier_resources import get_stop_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2

class Featurizer(object):

    def __init__(self):

        self.count_vect = CountVectorizer(ngram_range=(1, 1))
        self.tfidf_transformer = TfidfTransformer()
        self.best_feat = None

    def fit(self, queries, y):
        # tokenizing
        X_train_counts = self.count_vect.fit_transform(queries)
        list_index_words = {self.count_vect.vocabulary_[x]:x for x in self.count_vect.vocabulary_} 

        # from occurences to frequencies
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        # select best features
        stop_words = get_stop_words("en")
            
        # select best features with p value under threshold
        chi2val, pval = chi2(X_train_tfidf, y)
        self.best_feat = [i for i,v in enumerate(pval) if v < 0.4]
        if len(best_feat) == 0: # if no feature is selected, select min pvalue
            self.best_feat = [idx for idx, val in enumerate(pval) if val == pval.min()] 

        feature_names = {}
        for i in self.best_feat:
            feature_names[i] = { 'word': list_index_words[i] , 'pval': pval[i] }
            
        # remove stop words from features if their p-value is too high
        to_remove = []
        for feat in feature_names:
            if feature_names[feat]['word'] in stop_words:
                if feature_names[feat]['pval'] > 0.2:
                    self.best_feat.remove(feat)
                    to_remove.append(feat)
        
        return self

    def transform(self, queries):
        X_train_counts = self.count_vect.transform(queries)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        X = X_train_tfidf[:, self.best_feat]
        return best_feat, X


    def fit_transform(self, queries, y):
        self = self.fit(queries, y)
        return self.transform(queries)

import cPickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2

from intent_classifier_resources import get_stop_words
from snips_nlu.languages import Language
from snips_nlu.utils import ensure_string


class Featurizer(object):
    def __init__(self, language,
                 count_vectorizer=CountVectorizer(ngram_range=(1, 1)),
                 tfidf_transformer=TfidfTransformer(), pvalue_threshold=0.4):
        self.count_vectorizer = count_vectorizer
        self.tfidf_transformer = tfidf_transformer
        self.best_features = None
        self.pvalue_threshold = pvalue_threshold
        self.language = language

    def fit(self, queries, y):
        X_train_counts = self.count_vectorizer.fit_transform(queries)
        list_index_words = {self.count_vectorizer.vocabulary_[x]: x for x in
                            self.count_vectorizer.vocabulary_}
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        stop_words = get_stop_words(self.language)

        chi2val, pval = chi2(X_train_tfidf, y)
        self.best_features = [i for i, v in enumerate(pval) if
                              v < self.pvalue_threshold]
        if len(self.best_features) == 0:
            self.best_features = [idx for idx, val in enumerate(pval) if
                                  val == pval.min()]

        feature_names = {}
        for i in self.best_features:
            feature_names[i] = {'word': list_index_words[i], 'pval': pval[i]}

        for feat in feature_names:
            if feature_names[feat]['word'] in stop_words:
                if feature_names[feat]['pval'] > self.pvalue_threshold / 2.0:
                    self.best_features.remove(feat)

        return self

    def transform(self, queries):
        X_train_counts = self.count_vectorizer.transform(queries)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        X = X_train_tfidf[:, self.best_features]
        return X

    def fit_transform(self, queries, y):
        return self.fit(queries, y).transform(queries)

    def to_dict(self):
        return {
            'language_code': self.language.iso_code,
            'count_vectorizer': cPickle.dumps(self.count_vectorizer),
            'tfidf_transformer': cPickle.dumps(self.tfidf_transformer),
            'best_features': self.best_features,
            'pvalue_threshold': self.pvalue_threshold
        }

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict['count_vectorizer'] = ensure_string(
            obj_dict['count_vectorizer'])
        obj_dict['tfidf_transformer'] = ensure_string(
            obj_dict['tfidf_transformer'])
        self = cls(
            language=Language.from_iso_code(obj_dict['language_code']),
            count_vectorizer=cPickle.loads(obj_dict['count_vectorizer']),
            tfidf_transformer=cPickle.loads(obj_dict['tfidf_transformer']),
            pvalue_threshold=obj_dict['pvalue_threshold']
        )
        self.best_features = obj_dict["best_features"]
        return self

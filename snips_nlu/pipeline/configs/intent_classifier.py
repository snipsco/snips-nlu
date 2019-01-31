from __future__ import unicode_literals

from snips_nlu.common.from_dict import FromDict
from snips_nlu.constants import (
    CUSTOM_ENTITY_PARSER_USAGE, NOISE, STEMS, STOP_WORDS, WORD_CLUSTERS)
from snips_nlu.entity_parser.custom_entity_parser import (
    CustomEntityParserUsage)
from snips_nlu.pipeline.configs import Config, ProcessingUnitConfig
from snips_nlu.resources import merge_required_resources


class LogRegIntentClassifierConfig(FromDict, ProcessingUnitConfig):
    """Configuration of a :class:`.LogRegIntentClassifier`"""

    # pylint: disable=line-too-long
    def __init__(self, data_augmentation_config=None, featurizer_config=None,
                 random_seed=None):
        """
        Args:
            data_augmentation_config (:class:`IntentClassifierDataAugmentationConfig`):
                    Defines the strategy of the underlying data augmentation
            featurizer_config (:class:`FeaturizerConfig`): Configuration of the
                :class:`.Featurizer` used underneath
            random_seed (int, optional): Allows to fix the seed ot have
                reproducible trainings
        """
        if data_augmentation_config is None:
            data_augmentation_config = IntentClassifierDataAugmentationConfig()
        if featurizer_config is None:
            featurizer_config = FeaturizerConfig()
        self._data_augmentation_config = None
        self.data_augmentation_config = data_augmentation_config
        self._featurizer_config = None
        self.featurizer_config = featurizer_config
        self.random_seed = random_seed

    # pylint: enable=line-too-long

    @property
    def data_augmentation_config(self):
        return self._data_augmentation_config

    @data_augmentation_config.setter
    def data_augmentation_config(self, value):
        if isinstance(value, dict):
            self._data_augmentation_config = \
                IntentClassifierDataAugmentationConfig.from_dict(value)
        elif isinstance(value, IntentClassifierDataAugmentationConfig):
            self._data_augmentation_config = value
        else:
            raise TypeError("Expected instance of "
                            "IntentClassifierDataAugmentationConfig or dict"
                            "but received: %s" % type(value))

    @property
    def featurizer_config(self):
        return self._featurizer_config

    @featurizer_config.setter
    def featurizer_config(self, value):
        if isinstance(value, dict):
            self._featurizer_config = \
                FeaturizerConfig.from_dict(value)
        elif isinstance(value, FeaturizerConfig):
            self._featurizer_config = value
        else:
            raise TypeError("Expected instance of FeaturizerConfig or dict"
                            "but received: %s" % type(value))

    @property
    def unit_name(self):
        from snips_nlu.intent_classifier import LogRegIntentClassifier
        return LogRegIntentClassifier.unit_name

    def get_required_resources(self):
        resources = self.data_augmentation_config.get_required_resources()
        resources = merge_required_resources(
            resources, self.featurizer_config.get_required_resources())
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "data_augmentation_config":
                self.data_augmentation_config.to_dict(),
            "featurizer_config": self.featurizer_config.to_dict(),
            "random_seed": self.random_seed
        }


class IntentClassifierDataAugmentationConfig(FromDict, Config):
    """Configuration used by a :class:`.LogRegIntentClassifier` which defines
        how to augment data to improve the training of the classifier"""

    def __init__(self, min_utterances=20, noise_factor=5,
                 add_builtin_entities_examples=True, unknown_word_prob=0,
                 unknown_words_replacement_string=None,
                 max_unknown_words=None):
        """
        Args:
            min_utterances (int, optional): The minimum number of utterances to
                automatically generate for each intent, based on the existing
                utterances. Default is 20.
            noise_factor (int, optional): Defines the size of the noise to
                generate to train the implicit *None* intent, as a multiplier
                of the average size of the other intents. Default is 5.
            add_builtin_entities_examples (bool, optional): If True, some
                builtin entity examples will be automatically added to the
                training data. Default is True.
        """
        self.min_utterances = min_utterances
        self.noise_factor = noise_factor
        self.add_builtin_entities_examples = add_builtin_entities_examples
        self.unknown_word_prob = unknown_word_prob
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string
        if max_unknown_words is not None and max_unknown_words < 0:
            raise ValueError("max_unknown_words must be None or >= 0")
        self.max_unknown_words = max_unknown_words
        if unknown_word_prob > 0 and unknown_words_replacement_string is None:
            raise ValueError("unknown_word_prob is positive (%s) but the "
                             "replacement string is None" % unknown_word_prob)

    @staticmethod
    def get_required_resources():
        return {
            NOISE: True,
            STOP_WORDS: True
        }

    def to_dict(self):
        return {
            "min_utterances": self.min_utterances,
            "noise_factor": self.noise_factor,
            "add_builtin_entities_examples":
                self.add_builtin_entities_examples,
            "unknown_word_prob": self.unknown_word_prob,
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string,
            "max_unknown_words": self.max_unknown_words
        }


class FeaturizerConfig(FromDict, ProcessingUnitConfig):
    """Configuration of a :class:`.Featurizer` object"""

    # pylint: disable=line-too-long
    def __init__(self, tfidf_vectorizer_config=None,
                 cooccurrence_vectorizer_config=None,
                 pvalue_threshold=0.4,
                 added_cooccurrence_feature_ratio=0):
        """
        Args:
            tfidf_vectorizer_config (:class:`.TfidfVectorizerConfig`, optional):
                empty configuration of the featurizer's
                :attr:`tfidf_vectorizer`
            cooccurrence_vectorizer_config: (:class:`.CooccurrenceVectorizerConfig`, optional):
                configuration of the featurizer's
                :attr:`cooccurrence_vectorizer`
            pvalue_threshold (float): after fitting the training set to
                extract tfidf features, a univariate feature selection is
                applied. Features are tested for independence using a Chi-2
                test, under the null hypothesis that each feature should be
                equally present in each class. Only features having a p-value
                lower than the threshold are kept
            added_cooccurrence_feature_ratio (float, optional): proportion of
                cooccurrence features to add with respect to the number of
                tfidf features. For instance with a ratio of 0.5, if 100 tfidf
                features are remaining after feature selection, a maximum of 50
                cooccurrence features will be added
        """
        self.pvalue_threshold = pvalue_threshold
        self.added_cooccurrence_feature_ratio = \
            added_cooccurrence_feature_ratio

        if tfidf_vectorizer_config is None:
            tfidf_vectorizer_config = TfidfVectorizerConfig()
        elif isinstance(tfidf_vectorizer_config, dict):
            tfidf_vectorizer_config = TfidfVectorizerConfig.from_dict(
                tfidf_vectorizer_config)
        self.tfidf_vectorizer_config = tfidf_vectorizer_config

        if cooccurrence_vectorizer_config is None:
            cooccurrence_vectorizer_config = CooccurrenceVectorizerConfig()
        elif isinstance(cooccurrence_vectorizer_config, dict):
            cooccurrence_vectorizer_config = CooccurrenceVectorizerConfig \
                .from_dict(cooccurrence_vectorizer_config)
        self.cooccurrence_vectorizer_config = cooccurrence_vectorizer_config

    # pylint: enable=line-too-long

    @property
    def unit_name(self):
        from snips_nlu.intent_classifier import Featurizer
        return Featurizer.unit_name

    def get_required_resources(self):
        required_resources = self.tfidf_vectorizer_config \
            .get_required_resources()
        if self.cooccurrence_vectorizer_config:
            required_resources = merge_required_resources(
                required_resources,
                self.cooccurrence_vectorizer_config.get_required_resources())
        return required_resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "pvalue_threshold": self.pvalue_threshold,
            "added_cooccurrence_feature_ratio":
                self.added_cooccurrence_feature_ratio,
            "tfidf_vectorizer_config": self.tfidf_vectorizer_config.to_dict(),
            "cooccurrence_vectorizer_config":
                self.cooccurrence_vectorizer_config.to_dict(),
        }


class TfidfVectorizerConfig(FromDict, ProcessingUnitConfig):
    """Configuration of a :class:`.TfidfVectorizerConfig` object"""

    def __init__(self, word_clusters_name=None, use_stemming=False):
        """
        Args:
            word_clusters_name (str, optional): if a word cluster name is
                provided then the featurizer will use the word clusters IDs
                detected in the utterances and add them to the utterance text
                before computing the tfidf. Default to None
            use_stemming (bool, optional): use stemming before computing the
                tfdif. Defaults to False (no stemming used)
        """
        self.word_clusters_name = word_clusters_name
        self.use_stemming = use_stemming

    @property
    def unit_name(self):
        from snips_nlu.intent_classifier import TfidfVectorizer
        return TfidfVectorizer.unit_name

    def get_required_resources(self):
        resources = {STEMS: True if self.use_stemming else False}
        if self.word_clusters_name:
            resources[WORD_CLUSTERS] = {self.word_clusters_name}
        return resources

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "word_clusters_name": self.word_clusters_name,
            "use_stemming": self.use_stemming
        }


class CooccurrenceVectorizerConfig(FromDict, ProcessingUnitConfig):
    """Configuration of a :class:`.CooccurrenceVectorizer` object"""

    def __init__(self, window_size=None, unknown_words_replacement_string=None,
                 filter_stop_words=True, keep_order=True):
        """
        Args:
            window_size (int, optional): if provided, word cooccurrences will
                be taken into account only in a context window of size
                :attr:`window_size`. If the window size is 3 then given a word
                w[i], the vectorizer will only extract the following pairs:
                (w[i], w[i + 1]), (w[i], w[i + 2]) and (w[i], w[i + 3]).
                Defaults to None, which means that we consider all words
            unknown_words_replacement_string (str, optional)
            filter_stop_words (bool, optional): if True, stop words are ignored
                when computing cooccurrences
            keep_order (bool, optional): if True then cooccurrence are computed
                taking the words order into account, which means the pairs
                (w1, w2) and (w2, w1) will count as two separate features.
                Defaults to `True`.
        """
        self.window_size = window_size
        self.unknown_words_replacement_string = \
            unknown_words_replacement_string
        self.filter_stop_words = filter_stop_words
        self.keep_order = keep_order

    @property
    def unit_name(self):
        from snips_nlu.intent_classifier import CooccurrenceVectorizer
        return CooccurrenceVectorizer.unit_name

    def get_required_resources(self):
        return {
            STOP_WORDS: self.filter_stop_words,
            # We require the parser to be trained without stems because we
            # don't normalize and stem when processing in the
            # CooccurrenceVectorizer (in order to run the builtin and
            # custom parser on the same unormalized input).
            # Requiring no stems ensures we'll be able to parse the unstemmed
            # input
            CUSTOM_ENTITY_PARSER_USAGE: CustomEntityParserUsage.WITHOUT_STEMS
        }

    def to_dict(self):
        return {
            "unit_name": self.unit_name,
            "unknown_words_replacement_string":
                self.unknown_words_replacement_string,
            "window_size": self.window_size,
            "filter_stop_words": self.filter_stop_words,
            "keep_order": self.keep_order
        }

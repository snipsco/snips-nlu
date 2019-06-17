from sklearn.linear_model import SGDClassifier

from snips_nlu.constants import LANGUAGE, DATA, RES_INTENT_NAME, RES_PROBA
from snips_nlu.dataset import get_text_from_chunks, validate_and_format_dataset
from snips_nlu.intent_classifier.log_reg_classifier_utils import \
    build_training_data, get_regularization_factor
from snips_nlu.pipeline.configs import LogRegIntentClassifierConfig, \
    IntentClassifierDataAugmentationConfig
from snips_nlu.pipeline.processing_unit import ProcessingUnit

LOG_REG_ARGS = {
    "loss": "log",
    "penalty": "l2",
    "class_weight": "balanced",
    "max_iter": 1000,
    "tol": 1e-3,
    "n_jobs": -1
}


@ProcessingUnit.register("ood_detector")
class OODDetector(ProcessingUnit):
    def __init__(self, config=None, **shared):
        super(OODDetector, self).__init__(config, **shared)
        self.intent_classifier = None
        self.ood_classifier = None

    @property
    def fitted(self):
        return self.ood_classifier is not None

    def fit(self, dataset):
        from snips_nlu.intent_classifier import LogRegIntentClassifier

        dataset = validate_and_format_dataset(dataset)
        self.load_resources_if_needed(dataset[LANGUAGE])
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)
        classifier_config = LogRegIntentClassifierConfig(
            data_augmentation_config=IntentClassifierDataAugmentationConfig(
                noise_factor=0.0))
        self.intent_classifier = LogRegIntentClassifier(classifier_config)
        self.intent_classifier.fit(dataset)

        data_augmentation_config = IntentClassifierDataAugmentationConfig()
        utterances, classes, _ = build_training_data(
            dataset, dataset[LANGUAGE], data_augmentation_config,
            self.resources, self.random_state)
        noise_class = max(classes)
        classes = [1 if c == noise_class else 0 for c in classes]
        text_utterances = [get_text_from_chunks(u[DATA]) for u in utterances]
        X = self.transform(text_utterances)
        alpha = get_regularization_factor(dataset)
        self.ood_classifier = SGDClassifier(
            random_state=self.random_state, alpha=alpha, **LOG_REG_ARGS)
        self.ood_classifier.fit(X, classes)
        return self

    def transform(self, texts):
        scores = []
        for text in texts:
            intents = self.intent_classifier.get_intents(text)
            intents = [res for res in intents
                       if res[RES_INTENT_NAME] is not None]
            intents = sorted(intents, key=lambda res: res[RES_INTENT_NAME])
            scores.append([res[RES_PROBA] for res in intents])
        return scores

    def is_ood(self, text):
        X = self.transform([text])
        return self.ood_classifier._predict_proba(X)[0][1]

    def persist(self, path):
        pass

    @classmethod
    def from_path(cls, path, **shared):
        pass

from intent_parser import IntentParser
from snips_nlu.constants import (DATA, INTENTS, CUSTOM_ENGINE)
from snips_nlu.dataset import filter_dataset
from snips_nlu.languages import Language
from snips_nlu.slot_filler.crf_tagger import CRFTagger
from snips_nlu.slot_filler.crf_utils import (tags_to_slots,
                                             utterance_to_sample)
from snips_nlu.slot_filler.data_augmentation import augment_utterances
from snips_nlu.tokenization import tokenize
from snips_nlu.utils import (instance_to_generic_dict, instance_from_dict,
                             namedtuple_with_defaults)

_DataAugmentationConfig = namedtuple_with_defaults(
    '_DataAugmentationConfig',
    'max_utterances noise_prob min_noise_size max_noise_size',
    {
        'max_utterances': 0,
        'noise_prob': 0.,
        'min_noise_size': 0,
        'max_noise_size': 0
    }
)


class DataAugmentationConfig(_DataAugmentationConfig):
    def to_dict(self):
        return self._asdict()

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)


def default_data_augmentation_config(language):
    if language == Language.EN:
        return DataAugmentationConfig(max_utterances=200, noise_prob=0.05,
                                      min_noise_size=1, max_noise_size=3)
    else:
        return DataAugmentationConfig()


class ProbabilisticIntentParser(IntentParser):
    def __init__(self, language, intent_classifier, crf_taggers,
                 slot_name_to_entity_mapping, data_augmentation_config=None):
        super(ProbabilisticIntentParser, self).__init__()
        self.language = language
        self.intent_classifier = intent_classifier
        self._crf_taggers = None
        self.crf_taggers = crf_taggers
        self.slot_name_to_entity_mapping = slot_name_to_entity_mapping
        if data_augmentation_config is None:
            data_augmentation_config = default_data_augmentation_config(
                self.language)
        self.data_augmentation_config = data_augmentation_config

    @property
    def crf_taggers(self):
        return self._crf_taggers

    @crf_taggers.setter
    def crf_taggers(self, value):
        if any(t.language != self.language for t in value.values()):
            raise ValueError("Found taggers with different languages")
        self._crf_taggers = value

    def get_intent(self, text):
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_intent` is called")
        return self.intent_classifier.get_intent(text)

    def get_slots(self, text, intent=None):
        if intent is None:
            raise ValueError("intent can't be None")
        if not self.fitted:
            raise ValueError("ProbabilisticIntentParser must be fitted before "
                             "`get_slots` is called")
        if intent not in self.crf_taggers:
            raise KeyError("Invalid intent '%s'" % intent)

        tokens = tokenize(text)
        if len(tokens) == 0:
            return []
        intent_slots_mapping = self.slot_name_to_entity_mapping[intent]
        tagger = self.crf_taggers[intent]
        tags = tagger.get_tags(tokens)
        slots = tags_to_slots(text, tokens, tags, tagger.tagging_scheme,
                              intent_slots_mapping)
        return slots

    @property
    def fitted(self):
        return self.intent_classifier.fitted and all(
            slot_filler.fitted for slot_filler in self.crf_taggers.values())

    def fit(self, dataset):
        custom_dataset = filter_dataset(dataset, CUSTOM_ENGINE)
        self.intent_classifier = self.intent_classifier.fit(dataset)
        for intent_name in custom_dataset[INTENTS]:
            augmented_intent_utterances = augment_utterances(
                dataset, intent_name, language=self.language,
                **self.data_augmentation_config.to_dict())
            tagging_scheme = self.crf_taggers[intent_name].tagging_scheme
            crf_samples = [utterance_to_sample(u[DATA], tagging_scheme)
                           for u in augmented_intent_utterances]
            self.crf_taggers[intent_name] = self.crf_taggers[intent_name].fit(
                crf_samples)
        return self

    def to_dict(self):
        obj_dict = instance_to_generic_dict(self)
        obj_dict.update({
            "language_code": self.language.iso_code,
            "intent_classifier": self.intent_classifier.to_dict(),
            "crf_taggers": {intent_name: tagger.to_dict() for
                            intent_name, tagger in
                            self.crf_taggers.iteritems()},
            "slot_name_to_entity_mapping": self.slot_name_to_entity_mapping,
            "data_augmentation_config": self.data_augmentation_config.to_dict()
        })
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(
            language=Language.from_iso_code(obj_dict["language_code"]),
            intent_classifier=instance_from_dict(
                obj_dict["intent_classifier"]),
            crf_taggers={intent_name: CRFTagger.from_dict(tagger_dict)
                         for intent_name, tagger_dict in
                         obj_dict["crf_taggers"].iteritems()},
            slot_name_to_entity_mapping=obj_dict[
                "slot_name_to_entity_mapping"],
            data_augmentation_config=DataAugmentationConfig.from_dict(
                obj_dict["data_augmentation_config"])
        )

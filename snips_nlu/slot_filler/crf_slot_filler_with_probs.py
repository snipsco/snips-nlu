from __future__ import print_function
from __future__ import unicode_literals

import base64
import io
import math
import os
import tempfile
from builtins import range
from copy import copy
from itertools import groupby, permutations, product

from future.utils import iteritems
from sklearn_crfsuite import CRF

from snips_nlu.builtin_entities import is_builtin_entity, get_builtin_entities
from snips_nlu.constants import (
    RES_MATCH_RANGE, LANGUAGE, DATA, RES_ENTITY, START, END, RES_VALUE,
    ENTITY_KIND)
from snips_nlu.data_augmentation import augment_utterances
from snips_nlu.dataset import validate_and_format_dataset
from snips_nlu.pipeline.configs import CRFSlotFillerWithProbsConfig
from snips_nlu.preprocessing import stem
from snips_nlu.slot_filler import CRFSlotFiller
from snips_nlu.slot_filler.crf_slot_filler import _decode_tag
from snips_nlu.slot_filler.crf_slot_filler import _replace_builtin_tags
from snips_nlu.slot_filler.crf_utils import (
    TOKENS, TAGS, OUTSIDE, tags_to_slots, tag_name_to_slot_name,
    tags_to_preslots, positive_tagging, utterance_to_sample)
from snips_nlu.slot_filler.feature import TOKEN_NAME
from snips_nlu.slot_filler.feature_factory import get_feature_factory
from snips_nlu.slot_filler.slot_filler import SlotFiller
#from snips_nlu.tokenization import Token, tokenize
from snips_nlu.preprocessing import Token, tokenize
from snips_nlu.utils import (
    UnupdatableDict, mkdir_p, check_random_state, get_slot_name_mapping,
    ranges_overlap, NotTrained)


class CRFSlotFillerWithProbs(CRFSlotFiller):
    unit_name = "crf_slot_filler_with_probs"
    config_type = CRFSlotFillerWithProbsConfig

    def get_slots(self, text):
        """Extracts slots from the provided text

        Returns:
            list of dict: The list of extracted slots

        Raises:
            NotTrained: When the slot filler is not fitted
        """
        if not self.fitted:
            raise NotTrained("CRFSlotFiller must be fitted")
        tokens = tokenize(text, self.language)
        if not tokens:
            return []
        features = self.compute_features(tokens)
        tags = [_decode_tag(tag) for tag in
                self.crf_model.predict_single(features)]
        import operator
        tags_probs = []
        for t in self.crf_model.predict_marginals_single(features):
            max_entry = max(t.iteritems(), key=operator.itemgetter(1))
            tags_probs.append((_decode_tag(max_entry[0]), max_entry[1]))

        slots = tags_to_slots(text, tokens, tags_probs, self.config.tagging_scheme,
                              self.slot_name_mapping, with_probs=True)

        builtin_slots_names = set(slot_name for (slot_name, entity) in
                                  iteritems(self.slot_name_mapping)
                                  if is_builtin_entity(entity))
        if not builtin_slots_names:
            return slots

        # Replace tags corresponding to builtin entities by outside tags
        tags = _replace_builtin_tags(tags, builtin_slots_names)
        tags_probs = zip(tags, zip(*tags_probs)[1])
        return self._augment_slots(text, tokens, tags_probs, builtin_slots_names, with_probs=True)

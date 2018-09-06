from abc import ABCMeta, abstractproperty

from future.utils import with_metaclass

from snips_nlu.constants import (
    BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER, CUSTOM_ENTITY_PARSER_USAGE)
from snips_nlu.entity_parser.builtin_entity_parser import BuiltinEntityParser
from snips_nlu.entity_parser.custom_entity_parser import CustomEntityParser
from snips_nlu.entity_parser.custom_entity_parser_usage import \
    CustomEntityParserUsage
from snips_nlu.pipeline.configs import MLUnitConfig
from snips_nlu.pipeline.processing_unit import SerializableUnit, _get_unit_type
from snips_nlu.utils import classproperty


class MLUnit(with_metaclass(ABCMeta, SerializableUnit)):
    """ML pipeline unit

    ML processing units such as intent parsers, intent classifiers and
    slot fillers must implement this class.

    A :class:`MLUnit` is associated with a *config_type*, which
    represents the :class:`.MLUnitConfig` used to initialize it.
    """

    def __init__(self, config, **shared):
        if config is None or isinstance(config, MLUnitConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.config_type.from_dict(config)
        else:
            raise ValueError("Unexpected config type: %s" % type(config))
        self.builtin_entity_parser = shared.get(BUILTIN_ENTITY_PARSER)
        self.custom_entity_parser = shared.get(CUSTOM_ENTITY_PARSER)

    @classproperty
    def config_type(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError

    @abstractproperty
    def fitted(self):
        """Whether or not the processing unit has already been trained"""
        pass

    def fit_builtin_entity_parser_if_needed(self, dataset):
        # We fit an entity parser only if the unit has already been fitted
        # on a dataset, in this case we want to refit. We also if the parser
        # is none.
        # In the other case the parser is provided fitted by another unit
        if self.builtin_entity_parser is None or self.fitted:
            self.builtin_entity_parser = BuiltinEntityParser.build(
                dataset=dataset)
        return self

    def fit_custom_entity_parser_if_needed(self, dataset):
        # We only fit a custom entity parser when the unit has already been
        # fitted or if the parser is none.
        # In the other cases the parser is provided fitted by another unit.
        required_resources = self.config.get_required_resources()
        if not required_resources or not required_resources.get(
                CUSTOM_ENTITY_PARSER_USAGE):
            # In these cases we need a custom entity parser only to do the
            # final slot resolution step, which must be done without stemming.
            parser_usage = CustomEntityParserUsage.WITHOUT_STEMS
        else:
            parser_usage = required_resources[CUSTOM_ENTITY_PARSER_USAGE]

        if self.custom_entity_parser is None or self.fitted:
            self.custom_entity_parser = CustomEntityParser.build(
                dataset, parser_usage)
        return self


def get_ml_unit_config(unit_config):
    """Returns the :class:`.MLUnitConfig` corresponding to
        *unit_config*"""
    if isinstance(unit_config, MLUnitConfig):
        return unit_config
    elif isinstance(unit_config, dict):
        unit_name = unit_config["unit_name"]
        processing_unit_type = _get_unit_type(unit_name)
        return processing_unit_type.config_type.from_dict(unit_config)
    else:
        raise ValueError("Expected `unit_config` to be an instance of "
                         "MLUnitConfig or dict but found: %s"
                         % type(unit_config))


def build_ml_unit(unit_config, **shared):
    """Creates a new :class:`ProcessingUnit` from the provided *unit_config*

    Args:
        unit_config (:class:`.MLUnitConfig`): The processing unit
            config
        shared (kwargs): attributes shared across the NLU pipeline such as
            'builtin_entity_parser'.
    """
    unit = _get_unit_type(unit_config.unit_name)
    return unit(unit_config, **shared)

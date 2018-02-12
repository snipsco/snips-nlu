from abc import ABCMeta, abstractmethod
from builtins import object

from future.utils import with_metaclass

from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.utils import classproperty


class ProcessingUnit(with_metaclass(ABCMeta, object)):
    """Abstraction of a NLU pipeline unit

    Pipeline processing units such as intent parsers, intent classifiers and
    slot fillers must implement this class.

    A :class:`ProcessingUnit` is associated with a *config_type*, which
    represents the :class:`.ProcessingUnitConfig` used to initialize it.
    """

    def __init__(self, config):
        if config is None or isinstance(config, ProcessingUnitConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.config_type.from_dict(config)
        else:
            raise ValueError("Unexpected config type: %s" % type(config))

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError

    @classproperty
    def config_type(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, unit_dict):
        raise NotImplementedError


def _get_unit_type(unit_name):
    from snips_nlu.pipeline.units_registry import NLU_PROCESSING_UNITS

    unit = NLU_PROCESSING_UNITS.get(unit_name)
    if unit is None:
        raise ValueError("ProcessingUnit not found: %s" % unit_name)
    return unit


def get_processing_unit_config(unit_config):
    """Returns the :class:`.ProcessingUnitConfig` corresponding to
        *unit_config*"""
    if isinstance(unit_config, ProcessingUnitConfig):
        return unit_config
    elif isinstance(unit_config, dict):
        unit_name = unit_config["unit_name"]
        processing_unit_type = _get_unit_type(unit_name)
        return processing_unit_type.config_type.from_dict(unit_config)
    else:
        raise ValueError("Expected `unit_config` to be an instance of "
                         "ProcessingUnitConfig or dict but found: %s"
                         % type(unit_config))


def build_processing_unit(unit_config):
    """Creates a new :class:`ProcessingUnit` from the provided *unit_config*

    Args:
        unit_config (:class:`.ProcessingUnitConfig`): The processing unit
            config
    """
    unit = _get_unit_type(unit_config.unit_name)
    return unit(unit_config)


def load_processing_unit(unit_dict):
    """Load a :class:`ProcessingUnit` from a persisted processing unit dict"""
    unit = _get_unit_type(unit_dict["unit_name"])
    return unit.from_dict(unit_dict)

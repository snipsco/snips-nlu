from __future__ import unicode_literals

import json
import shutil
from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path

from future.builtins import object
from future.utils import with_metaclass

from snips_nlu.constants import (
    BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER, CUSTOM_ENTITY_PARSER_USAGE)
from snips_nlu.entity_parser import (
    BuiltinEntityParser, CustomEntityParser, CustomEntityParserUsage)
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.utils import classproperty, json_string, temp_dir, unzip_archive


class ProcessingUnit(with_metaclass(ABCMeta, object)):
    """Abstraction of a NLU pipeline unit

    Pipeline processing units such as intent parsers, intent classifiers and
    slot fillers must implement this class.

     A :class:`ProcessingUnit` is associated with a *config_type*, which
    represents the :class:`.ProcessingUnitConfig` used to initialize it.
    """

    def __init__(self, config, **shared):
        if config is None or isinstance(config, ProcessingUnitConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.config_type.from_dict(config)
        else:
            raise ValueError("Unexpected config type: %s" % type(config))
        self.builtin_entity_parser = shared.get(BUILTIN_ENTITY_PARSER)
        self.custom_entity_parser = shared.get(CUSTOM_ENTITY_PARSER)

    @abstractproperty
    def fitted(self):
        """Whether or not the processing unit has already been trained"""
        pass

    def fit_builtin_entity_parser_if_needed(self, dataset):
        # We only fit a builtin entity parser when the unit has already been
        # fitted or if the parser is none.
        # In the other cases the parser is provided fitted by another unit.
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

    def persist_metadata(self, path, **kwargs):
        metadata = {"unit_name": self.unit_name}
        metadata.update(kwargs)
        metadata_json = json_string(metadata)
        with (path / "metadata.json").open(mode="w") as f:
            f.write(metadata_json)

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError

    @classproperty
    def config_type(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError

    @abstractmethod
    def persist(self, path):
        pass

    @classmethod
    def from_path(cls, path, **shared):
        raise NotImplementedError

    def to_byte_array(self):
        """Serialize the :class:`ProcessingUnit` instance into a bytearray

        This method persists the processing unit in a temporary directory, zip
        the directory and return the zipped file as binary data.

        Returns:
            bytearray: the processing unit as bytearray data
        """
        cleaned_unit_name = _sanitize_unit_name(self.unit_name)
        with temp_dir() as tmp_dir:
            processing_unit_dir = tmp_dir / cleaned_unit_name
            self.persist(processing_unit_dir)
            archive_base_name = tmp_dir / cleaned_unit_name
            archive_name = archive_base_name.with_suffix(".zip")
            shutil.make_archive(
                base_name=str(archive_base_name), format="zip",
                root_dir=str(tmp_dir), base_dir=cleaned_unit_name)
            with archive_name.open(mode="rb") as f:
                processing_unit_bytes = bytearray(f.read())
        return processing_unit_bytes

    @classmethod
    def from_byte_array(cls, unit_bytes, **shared):
        """Load a :class:`ProcessingUnit` instance from a bytearray

        Args:
            unit_bytes (bytearray): A bytearray representing a zipped
                processing unit.
        """
        cleaned_unit_name = _sanitize_unit_name(cls.unit_name)
        with temp_dir() as tmp_dir:
            archive_path = (tmp_dir / cleaned_unit_name).with_suffix(".zip")
            with archive_path.open(mode="wb") as f:
                f.write(unit_bytes)
            unzip_archive(archive_path, str(tmp_dir))
            processing_unit = cls.from_path(tmp_dir / cleaned_unit_name,
                                            **shared)
        return processing_unit


def _sanitize_unit_name(unit_name):
    return unit_name \
        .lower() \
        .strip() \
        .replace(" ", "") \
        .replace("/", "") \
        .replace("\\", "")


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
        raise ValueError(
            "Expected `unit_config` to be an instance of ProcessingUnitConfig "
            "or dict but found: %s" % type(unit_config))


def build_processing_unit(unit_config, **shared):
    """Creates a new :class:`ProcessingUnit` from the provided *unit_config*

    Args:
        unit_config (:class:`.ProcessingUnitConfig`): The processing unit
            config
        shared (kwargs): attributes shared across the NLU pipeline such as
            'builtin_entity_parser'.
    """
    unit = _get_unit_type(unit_config.unit_name)
    return unit(unit_config, **shared)


def load_processing_unit(unit_path, **shared):
    """Load a :class:`ProcessingUnit` from a persisted processing unit
        directory"""
    unit_path = Path(unit_path)
    with (unit_path / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    unit = _get_unit_type(metadata["unit_name"])
    return unit.from_path(unit_path, **shared)

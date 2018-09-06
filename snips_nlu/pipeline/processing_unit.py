from __future__ import unicode_literals

import json
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path

from future.builtins import object
from future.utils import with_metaclass

from snips_nlu.utils import classproperty, json_string, temp_dir, unzip_archive


class ProcessingUnit(with_metaclass(ABCMeta, object)):
    """Abstraction of a NLU pipeline unit"""

    @classproperty
    def unit_name(cls):  # pylint:disable=no-self-argument
        raise NotImplementedError


class SerializableUnit(with_metaclass(ABCMeta, ProcessingUnit)):
    """Abstraction of a NLU pipeline unit"""

    @abstractmethod
    def persist(self, path):
        pass

    @classmethod
    def from_path(cls, path, **shared):
        raise NotImplementedError

    def persist_metadata(self, path, **kwargs):
        metadata = {"unit_name": self.unit_name}
        metadata.update(kwargs)
        metadata_json = json_string(metadata)
        with (path / "metadata.json").open(mode="w") as f:
            f.write(metadata_json)

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
            shutil.make_archive(base_name=str(archive_base_name),
                                format="zip", root_dir=str(tmp_dir),
                                base_dir=cleaned_unit_name)
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


def load_processing_unit(unit_path, **shared):
    """Load a :class:`ProcessingUnit` from a persisted processing unit
    directory"""
    unit_path = Path(unit_path)
    with (unit_path / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    unit = _get_unit_type(metadata["unit_name"])
    return unit.from_path(unit_path, **shared)

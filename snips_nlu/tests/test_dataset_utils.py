from __future__ import unicode_literals

import io
from unittest import TestCase

from snips_nlu.dataset import Dataset, validate_and_format_dataset
from snips_nlu.dataset.utils import extract_entity_values


class TestDatasetUtils(TestCase):
    def test_should_extract_entity_values(self):
        # Given
        set_light_color_yaml = io.StringIO("""
---
type: intent
name: setLightColor
utterances:
  - set the lights to [color](blue)
  - change the light to [color](yellow) in the [room](bedroom)""")

        turn_light_on_yaml = io.StringIO("""
---
type: intent
name: turnLightOn
utterances:
  - turn the light on in the [room](kitchen)
  - turn the [room](bathroom)'s lights on""")

        color_yaml = io.StringIO("""
type: entity
name: color
values:
- [blue, cyan]
- red""")

        room_yaml = io.StringIO("""
type: entity
name: room
values:
- garage
- [living room, main room]""")

        dataset_files = [set_light_color_yaml, turn_light_on_yaml, color_yaml,
                         room_yaml]
        dataset = Dataset.from_yaml_files("en", dataset_files).json
        dataset = validate_and_format_dataset(dataset)

        # When
        entity_values = extract_entity_values(dataset,
                                              apply_normalization=True)

        # Then
        expected_values = {
            "setLightColor": {"blue", "yellow", "cyan", "red", "bedroom",
                              "garage", "living room", "main room", "kitchen",
                              "bathroom"},
            "turnLightOn": {"bedroom", "garage", "living room", "main room",
                            "kitchen", "bathroom"}
        }
        self.assertDictEqual(expected_values, entity_values)

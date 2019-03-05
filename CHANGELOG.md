# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Fixed
- Issue with intent classification reducing classification accuracy

## [0.19.2] - 2019-02-11
### Fixed
- Fix an issue regarding the way builtin entities were handled by the `CRFSlotFiller`

## [0.19.1] - 2019-02-04
### Fixed
- Bug causing an unnecessary reloading of shared resources

## [0.19.0] - 2019-02-04
### Added
- Support for Python3.7
- `get_intents(text)` API in `SnipsNLUEngine` to get the probabilities of all the intents
- `get_slots(text, intent)` API in `SnipsNLUEngine` to extract slots when the intent is known
- The `DeterministicIntentParser` can now ignore stop words through the new `ignore_stop_words` configuration parameter
- Co-occurrence features can now be used in the `LogRegIntentClassifier`

### Changed
- Remove the deprecated text file format for intents and entities in favor YAML format
- The `None` intent is now handled as a regular intent in the parsing output, which means that:
```python
{
    "input": "foo bar",
    "intent": None,
    "slots": None
}
```

is replaced with:
 
```python
{
    "input": "foo bar",
    "intent": {
        "intentName": None,
        "probability": 0.552122
    },
    "slots": []
}
```
- Patterns of the `DeterministicIntentParser` are now deduplicated across intents in order to reduce ambiguity
- Improve the use of custom `ProcessingUnit` through the use of `Registrable` pattern
- Improve the use of default processing unit configurations 
- Improve logging
- Replace `snips-nlu-ontology` with `snips-nlu-parsers`
- Drop support for Python3.4

### Fixed
- Issue when persisting resources
- Issue when resolving custom entities
- Issue with whitespaces when generating dataset from YAML and text files
- Issue with unicode when using the CLI (Python 2)

## [0.18.0] - 2018-11-26
### Added
- New YAML format to create dataset
- Verbose mode in CLI

### Changed
- Bump `snips-nlu-ontology` to `0.62.0` to improve memory usage 


## [0.17.4] - 2018-11-20
### Added
- Add a `--config` argument in the metrics CLI

### Changed
- Replace "parser_threshold" by "matching_strictness" in dataset format
- Optimize loading and inference runtime
- Disable stemming for intent classification in default configs


## [0.17.3] - 2018-10-18
### Fixed
- Crash with num2words and floats

## [0.17.2] - 2018-10-15
### Added
- Support for builtin music entities in english

## [0.17.1] - 2018-10-09
### Fixed
- `DeterministicIntentParser` now relies on the custom entity parser

### Changed
- Bump `snips-nlu-ontology` to `0.60`

## [0.17.0] - 2018-10-05
### Added
- Support for 3 new builtin entities in French: `snips/musicAlbum`, `snips/musicArtist` and `snips/musicTrack`
- Minimal support for Italian

### Changed
- model version `0.16.0` => `0.17.0`

### Fixed
- Bug with entity feature name in intent classification

## [0.16.5] - 2018-09-06
### Fixed
- Segfault in CRFSuite when the `CRFSlotFiller` is fitted only on empty utterances 

## [0.16.4] - 2018-08-30
### Fixed
- Issue with the `CrfSlotFiller` file names in the `ProbabilisticIntentParser` serialization  

## [0.16.3] - 2018-08-22
### Fixed
- Issue with synonyms when multiple synonyms have the same normalization

## [0.16.2] - 2018-08-08
### Added
- `automatically_extensible` flag in dataset generation tool
- System requirements
- Reference to chatito tool in documentation

### Changed
- Bump `snips-nlu-ontology` to `0.57.3`
- versions of dependencies are now defined more loosely

### Fixed
- Issue with synonyms mapping
- Issue with `snips-nlu download-all-languages` CLI command

## [0.16.1] - 2018-07-23
### Added
- Every processing unit can be persisted into (and loaded from) a `bytearray`

## [0.16.0] - 2018-07-17
### Changed
- The `SnipsNLUEngine` object is now persisted to (and loaded from) a 
directory, instead of a single json file.
- The language resources are now persisted along with the `SnipsNLUEngine`, 
removing the need to download and load the resources when loading a trained engine.
- The format of language resources has been optimized.

### Added
- Stemmed gazetteers, computed beforehand. It removes the need to stem 
gazetteers on the fly.
- API to persist (and load) a `SnipsNLUEngine` object as a `bytearray`

### Fixed
- Issue in the `DeterministicIntentParser` when the same slot name was used in 
multiple intents while referring to different entities

## [0.15.1] - 2018-07-09
### Changed
- Bump `snips-nlu-ontology` to `0.57.1`

### Fixed
- Crash when parsing implicit years before 1970

## [0.15.0] - 2018-06-21
### Changed
- Language resources are now packaged separately from the Snips NLU core
library, and can be fetched using `snips-nlu download <language>`.
- The CLI tool now consists in a single entry point, `snips-nlu`, which exposes
several commands.

### Added
- CLI command to parse a query


## [0.14.0] - 2018-06-08
### Fixed
- Issue due to caching of builtin entities at inference time

### Changed
- Improve builtin entities handling during intent classification
- Improve builtin entities handling in `DeterministicIntentParser`
- Reduce size of regex patterns in trained model file
- Update model version to `0.15.0`

## [0.13.5] - 2018-05-23
### Fixed
- Fixed synonyms matching by using the normalized version of the tagged values
- Fixed dataset augmentation by keeping stripped values of entities
- Fixed the string variations functions to prevent generating too many variations   

## [0.13.4] - 2018-05-18
### Added
- Documentation for the `None` intent

### Changed
- Improve calibration of intent classification probabilities
- Update snips-nlu-ontology version to 0.55.0

### Fixed
- DeterministicIntentParser: Fix bug when deduplicating regexes
- DeterministicIntentParser: Fix issue with incorrect ranges when parsing sentences with both builtin and custom slots
- DeterministicIntentParser: Fix issue with builtin entities placeholders causing mismatches
- Fix issue with engine-inference CLI script not loading resources correctly 

## [0.13.3] - 2018-04-24
### Added
- Add config parameter to augment data with builtin entity examples

### Changed
- Bump snips-nlu-ontology to 0.54.3
- Use language specific configs by default
- Add right space to chunks in data augmentation
- Update JA config

### Fixed
- Fix inconsistency bug with shape ngram CRF feature
- Fix bug when initializing `CRFSlotFiller` with config dict
- Fix bug with gazetteer in ngram feature
- Fix bug with length CRF feature

## [0.13.1] - 2018-04-10
### Changed
- Bump ontology version from 0.54.1 to 0.54.2

### Fixed
- Fix CRF parsing of builtin entities by adding builtin entities examples of different length
- Fix CLI scripts importing metrics package which might not be installed

## [0.13.0] - 2018-04-06
### Added
- Add contributing guidelines, code of conduct, authors and contributors
- Add integration test
- Add CHANGELOG

### Changed
- Bump model version from 0.13.0 to 0.14
- Improve intent classification by leveraging builtin entities
- Improve loading of language specific resources
- Improve support of japanese

### Removed
- Remove `exhaustive_permutations_threshold` parameter in config

### Fixed
- Fix compiling issue with `bindgen` dependency when installing from source
- Fix issue in `CRFSlotFiller` when handling builtin entities

[Unreleased]: https://github.com/snipsco/snips-nlu/compare/0.19.2...HEAD
[0.19.2]: https://github.com/snipsco/snips-nlu/compare/0.19.1...0.19.2
[0.19.1]: https://github.com/snipsco/snips-nlu/compare/0.19.0...0.19.1
[0.19.0]: https://github.com/snipsco/snips-nlu/compare/0.18.0...0.19.0
[0.18.0]: https://github.com/snipsco/snips-nlu/compare/0.17.4...0.18.0
[0.17.4]: https://github.com/snipsco/snips-nlu/compare/0.17.3...0.17.4
[0.17.3]: https://github.com/snipsco/snips-nlu/compare/0.17.2...0.17.3
[0.17.2]: https://github.com/snipsco/snips-nlu/compare/0.17.1...0.17.2
[0.17.1]: https://github.com/snipsco/snips-nlu/compare/0.17.0...0.17.1
[0.17.0]: https://github.com/snipsco/snips-nlu/compare/0.16.5...0.17.0
[0.16.5]: https://github.com/snipsco/snips-nlu/compare/0.16.4...0.16.5
[0.16.4]: https://github.com/snipsco/snips-nlu/compare/0.16.3...0.16.4
[0.16.3]: https://github.com/snipsco/snips-nlu/compare/0.16.2...0.16.3
[0.16.2]: https://github.com/snipsco/snips-nlu/compare/0.16.1...0.16.2
[0.16.1]: https://github.com/snipsco/snips-nlu/compare/0.16.0...0.16.1
[0.16.0]: https://github.com/snipsco/snips-nlu/compare/0.15.1...0.16.0
[0.15.1]: https://github.com/snipsco/snips-nlu/compare/0.15.0...0.15.1
[0.15.0]: https://github.com/snipsco/snips-nlu/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/snipsco/snips-nlu/compare/0.13.5...0.14.0
[0.13.5]: https://github.com/snipsco/snips-nlu/compare/0.13.4...0.13.5
[0.13.4]: https://github.com/snipsco/snips-nlu/compare/0.13.3...0.13.4
[0.13.3]: https://github.com/snipsco/snips-nlu/compare/0.13.2...0.13.3
[0.13.1]: https://github.com/snipsco/snips-nlu/compare/0.13.0...0.13.1
[0.13.0]: https://github.com/snipsco/snips-nlu/compare/0.12.1...0.13.0

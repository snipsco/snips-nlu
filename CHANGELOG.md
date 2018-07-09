# Changelog
All notable changes to this project will be documented in this file.

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

[0.15.1]: https://github.com/snipsco/snips-nlu/compare/0.15.0...0.15.1
[0.15.0]: https://github.com/snipsco/snips-nlu/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/snipsco/snips-nlu/compare/0.13.5...0.14.0
[0.13.5]: https://github.com/snipsco/snips-nlu/compare/0.13.4...0.13.5
[0.13.4]: https://github.com/snipsco/snips-nlu/compare/0.13.3...0.13.4
[0.13.3]: https://github.com/snipsco/snips-nlu/compare/0.13.2...0.13.3
[0.13.1]: https://github.com/snipsco/snips-nlu/compare/0.13.0...0.13.1
[0.13.0]: https://github.com/snipsco/snips-nlu/compare/0.12.1...0.13.0

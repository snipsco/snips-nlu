# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

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

[Unreleased]: https://github.com/snipsco/snips-nlu/compare/0.13.0...HEAD
[0.13.0]: https://github.com/snipsco/snips-nlu/compare/0.12.1...0.13.0
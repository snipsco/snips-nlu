from snips_nlu.__about__ import __model_version__


class SnipsNLUError(Exception):
    """Base class for exceptions raised in the snips-nlu library"""


class IncompatibleModelError(Exception):
    """Raised when trying to load an incompatible NLU engine

    This happens when the engine data was persisted with a previous version of
    the library which is not compatible with the one used to load the model.
    """

    def __init__(self, persisted_version):
        super(IncompatibleModelError, self).__init__(
            "Incompatible data model: persisted model=%s, python lib model=%s"
            % (persisted_version, __model_version__))


class InvalidInputError(Exception):
    """Raised when an incorrect input is passed to one of the APIs"""


class NotTrained(SnipsNLUError):
    """Raised when a processing unit is used while not fitted"""


class IntentNotFoundError(SnipsNLUError):
    """Raised when an intent is used although it was not part of the
    training data"""

    def __init__(self, intent):
        super(IntentNotFoundError, self).__init__("Unknown intent '%s'"
                                                  % intent)


class DatasetFormatError(SnipsNLUError):
    """Raised when attempting to create a Snips NLU dataset using a wrong
    format"""


class _EmptyDatasetUtterancesError(SnipsNLUError):
    """Raised when attempting to train a processing unit on a dataset having
     only empty utterances"""


class EntityFormatError(DatasetFormatError):
    """Raised when attempting to create a Snips NLU entity using a wrong
    format"""


class IntentFormatError(DatasetFormatError):
    """Raised when attempting to create a Snips NLU intent using a wrong
    format"""


class AlreadyRegisteredError(SnipsNLUError):
    """Raised when attempting to register a subclass which is already
    registered"""

    def __init__(self, name, new_class, existing_class):
        msg = "Cannot register %s for %s as it has already been used to " \
              "register %s" \
              % (name, new_class.__name__, existing_class.__name__)
        super(AlreadyRegisteredError, self).__init__(msg)


class NotRegisteredError(SnipsNLUError):
    """Raised when trying to use a subclass which was not registered"""

    def __init__(self, registrable_cls, name=None, registered_cls=None):
        if name is not None:
            msg = "'%s' has not been registered for type %s. " \
                  % (name, registrable_cls)
        else:
            msg = "subclass %s has not been registered for type %s. " \
                  % (registered_cls, registrable_cls)
        msg += "Use @BaseClass.register('my_component') to register a subclass"
        super(NotRegisteredError, self).__init__(msg)


class PersistingError(SnipsNLUError):
    """Raised when trying to persist a processing unit to a path which already
    exists"""

    def __init__(self, path):
        super(PersistingError, self).__init__("Path already exists: %s"
                                              % str(path))


class LoadingError(SnipsNLUError):
    """Raised when trying to load a processing unit while some files are
    missing"""

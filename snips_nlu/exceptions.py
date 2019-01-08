class SnipsNLUError(Exception):
    """Base class for exceptions raised in the snips-nlu library"""


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

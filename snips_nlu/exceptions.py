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


class EntityFormatError(SnipsNLUError):
    """Raised when attempting to create a Snips NLU entity using a wrong
    format"""


class IntentFormatError(SnipsNLUError):
    """Raised when attempting to create a Snips NLU intent using a wrong
    format"""

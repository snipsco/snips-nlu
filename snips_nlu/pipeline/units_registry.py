from deprecation import deprecated

from snips_nlu.__about__ import __version__
from snips_nlu.common.registrable import Registrable


# pylint:disable=protected-access

@deprecated(deprecated_in="0.18.1", removed_in="0.19.0",
            current_version=__version__,
            details="Use the @BaseClass.register decorator instead")
def register_processing_unit(unit_type):
    """Allow to register a processing unit

    Raises:
        AlreadyRegisteredError: when trying to register a processing unit
            which has already been registered
        TypeError: when the unit to register does not inherit from a unit
            base class
    """
    bases = unit_type.__bases__
    for base_cls in bases:
        if base_cls in Registrable._registry:
            base_cls.register(unit_type.unit_name)
    base_names = [base_cls.__name__ for base_cls in Registrable._registry]
    raise TypeError("%s must inherit from one of the following base class in "
                    "order to be registered: %s"
                    % (unit_type.__name__, base_names))

# pylint:enable=protected-access

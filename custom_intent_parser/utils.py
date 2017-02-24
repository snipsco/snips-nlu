import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


def sequence_equal(seq, other_seq):
    return len(seq) == len(other_seq) and sorted(seq) == sorted(other_seq)

"""A simple pure python recursive trie implementation"""
import gzip


class SymbolTable(object):
    def __init__(self):
        self.symbols = {}

    def add_symbol(self, symbol):
        key = hash(symbol)
        self.symbols[key] = symbol
        return key

    def get_symbol(self, key):
        return self.symbols[key]

    def get_key(self, symbol):
        key = hash(symbol)
        if key in self.symbols:
            return key
        return None

    def write_to_file(self, filename):
        text = "\n".join(self.symbols.values())
        with gzip.open(filename, 'wb') as f:
                f.write(text.encode("utf-8"))

    @classmethod
    def from_file(cls, filename):
        instance = cls()
        with gzip.open(filename, 'r') as f:
            for symbol in f:
                instance.add_symbol(symbol)


class Trie(object):
    __slots__ = ('children', 'value', 'final')

    def __init__(self):
        self.children = {}
        self.value = None
        self.final = False

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def items(self):
        return list(self.iteritems())

    def iteritems(self, keys=None):
        for key, child in self.children.iteritems():
            child_keys = keys or []
            for x in child.iteritems(keys=child_keys+[key]):
                yield x
        if self.final:
            yield (keys, self.value)

    def write_to_file(self, filename):
        with gzip.open(filename, 'wb') as f:
            for line in self.text_stream():
                line = line + "\n"
                f.write(line.encode("utf-8"))

    @classmethod
    def from_file(cls):
        raise NotImplementedError

    def text_stream(self):
        for (key, val) in self.iteritems():
            yield "{}, {}".format(" ".join([str(x) for x in key]), " ".join([str(x) for x in val]))

    def __setitem__(self, key, value):
        node = self
        for frag in key:
            if frag in node.children:
                node = node.children[frag]
            else:
                new_node = Trie()
                node.children[frag] = new_node
                node = new_node
        node.value = value
        node.final = True

    def __getstate__(self):
        return {'children': self.children, 'value': self.value, 'final': self.final}

    def __getitem__(self, key):
        node = self
        for frag in key:
            node = node.children.get(frag)
            if not node:
                raise KeyError(key)

        if node.final:
            return node.value
        else:
            raise KeyError(key)

    def __eq__(self, other):
        return self.__getstate__() == other.__getstate__()


def test():

    trie = Trie()
    key, val = ("book a train %loc%".split(' '), "book::intent book a train %loc%(start)".split(' '))
    trie[key] = val
    print(trie.get(key))
    print(trie.items())


if __name__ == '__main__':
    test()

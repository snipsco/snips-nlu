try:
    import funcsigs as inspect
except ImportError:
    import inspect

from future.utils import iteritems

KEYWORD_KINDS = {inspect.Parameter.POSITIONAL_OR_KEYWORD,
                 inspect.Parameter.KEYWORD_ONLY}


class FromDict(object):
    @classmethod
    def from_dict(cls, dict):
        if dict is None:
            return cls()
        params = inspect.signature(cls.__init__).parameters

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in
               params.values()):
            return cls(**dict)

        param_names = set()
        for i, (name, param) in enumerate(iteritems(params)):
            if i == 0 and name == "self":
                continue
            if param.kind in KEYWORD_KINDS:
                param_names.add(name)
        filtered_dict = {k: v for k, v in iteritems(dict) if k in param_names}
        return cls(**filtered_dict)

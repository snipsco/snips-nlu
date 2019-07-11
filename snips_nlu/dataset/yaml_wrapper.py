import yaml


def _construct_yaml_str(self, node):
    # Override the default string handling function
    # to always return unicode objects
    return self.construct_scalar(node)


yaml.Loader.add_constructor("tag:yaml.org,2002:str", _construct_yaml_str)
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", _construct_yaml_str)

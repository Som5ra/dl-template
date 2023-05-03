
def get_config(config_file):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("module.name", config_file)
    config = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config
    spec.loader.exec_module(config)

    return config
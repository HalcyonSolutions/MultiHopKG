import argparse
import yaml
import os

def overload_parse_defaults_with_yaml(yaml_location:str, args: argparse.Namespace):
    # check if the yaml file exists
    if not os.path.exists(yaml_location):
        print(f"Yaml file {yaml_location} does not exist, skipping yaml overload")
        return args
    
    with open(yaml_location, "r") as f:
        print(f"Trying to import the yaml file {yaml_location}")
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Imported yam with keys {yaml_args.keys()}")
        overloaded_args = recurse_til_leaf(yaml_args)
        for k, v in overloaded_args.items():
            if k in args.__dict__:
                # args.__dict__[k] = v
                # Change the property not they key
                setattr(args, k, v)
            else:
                raise ValueError(f"Key {k} not found in args")
    return args

def recurse_til_leaf(d: dict, parent_key: str = "") -> dict:
    return_dict = {}
    for k, v in d.items():
        next_key = f"{parent_key}_{k}" if parent_key != "" else k
        if isinstance(v, dict):
            deep_dict = recurse_til_leaf(v, parent_key=next_key)
            return_dict.update(deep_dict)
        else:
            return_dict[next_key] = v
    return return_dict

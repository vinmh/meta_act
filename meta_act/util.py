from typing import Dict, Any


def dict_str_creator(obj: Dict[Any, Any]):
    return ", ".join([f"{k}: {v}" for k, v in obj.items()])

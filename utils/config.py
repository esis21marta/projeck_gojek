import os
from typing import Dict, Any
import sys
import toml

def load_config() -> Dict[str, Any]:
    filepath = "config.toml"
    with open(filepath, "r") as f:
        return toml.load(f)
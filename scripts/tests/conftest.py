"""Load the tooling package before pytest 7 importlib creates dummy parents."""

import importlib

importlib.import_module("scripts")

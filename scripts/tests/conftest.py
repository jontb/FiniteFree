"""Load source packages before pytest 7 importlib creates dummy parents."""

import importlib

importlib.import_module("scripts")
importlib.import_module("tests")

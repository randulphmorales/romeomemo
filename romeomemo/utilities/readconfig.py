#!/usr/bin/env python
# coding=utf-8

import os
import sys

from importlib import import_module

def load_cfg(cfg_path):
    """Load config file"""
    # Remove a (possible) trailing file-extension from the config path
    # (importlib doesn't want it)
    cfg_path = os.path.splitext(cfg_path)[0]
    try:
        sys.path.append(os.path.dirname(os.path.realpath(cfg_path)))
        cfg = import_module(os.path.basename(cfg_path))
    except IndexError:
        raise FileNotFoundError(
            f"Provided config file {cfg_path} doesn't exist"
        )

    return cfg



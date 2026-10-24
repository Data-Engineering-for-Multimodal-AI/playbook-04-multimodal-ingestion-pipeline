#!/usr/bin/env python3
"""Runs ByteEats/extractors/video_extractor.py, the one copy of this script.

Kept so the earlier command `python video_feature_extractor.py ...` still works; it takes the same arguments.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ByteEats', 'extractors', 'video_extractor.py'), run_name="__main__")

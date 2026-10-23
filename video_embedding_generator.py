#!/usr/bin/env python3
"""Runs ByteEats/analysis/embedding_generator.py, the one copy of this script.

Kept so the earlier command `python video_embedding_generator.py ...` still works; it takes the same arguments.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ByteEats', 'analysis', 'embedding_generator.py'), run_name="__main__")

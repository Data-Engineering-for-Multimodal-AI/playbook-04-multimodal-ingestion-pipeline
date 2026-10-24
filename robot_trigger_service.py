#!/usr/bin/env python3
"""Runs ByteEats/services/robot_trigger.py, the one copy of this script.

Kept so the earlier command `python robot_trigger_service.py ...` still works; it takes the same arguments.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ByteEats', 'services', 'robot_trigger.py'), run_name="__main__")

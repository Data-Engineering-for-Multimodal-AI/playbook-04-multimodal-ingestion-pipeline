#!/usr/bin/env python3
"""Runs ByteEats/services/notification.py, the one copy of this script.

Kept so the earlier command `python pickup_notification_service.py ...` still works; it takes the same arguments.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ByteEats', 'services', 'notification.py'), run_name="__main__")

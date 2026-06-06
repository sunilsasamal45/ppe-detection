"""
WSGI config for PPE Detection System.

This module contains the WSGI application used by Django's runserver.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ppe.settings')

application = get_wsgi_application()

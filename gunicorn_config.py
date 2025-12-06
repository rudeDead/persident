"""Gunicorn configuration for Render deployment"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Single worker for free tier memory limits
worker_class = 'sync'
worker_connections = 1000
threads = 2  # 2 threads per worker
timeout = 300  # 5 minutes for model loading
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Preload app to load models once before forking
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'molecular_prediction'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

def on_starting(server):
    """Called just before the master process is initialized."""
    print("="*60)
    print("Starting Gunicorn server...")
    print(f"Workers: {workers}")
    print(f"Threads per worker: {threads}")
    print(f"Timeout: {timeout}s")
    print("="*60)

def when_ready(server):
    """Called just after the server is started."""
    print("Server is ready. Awaiting requests.")

def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("Shutting down server...")

# gunicorn.conf.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
IGNORE_DIR = LOG_DIR / "ignore"

LOG_DIR.mkdir(parents=True, exist_ok=True)
IGNORE_DIR.mkdir(parents=True, exist_ok=True)

# Server
bind = "127.0.0.1:8002"
workers = 2
threads = 4
timeout = 60
worker_class = "gthread"
forwarded_allow_ips = "127.0.0.1"

# Logging
loglevel = "info"
capture_output = True

# Keep gunicorn startup/errors/prints here
errorlog = str(LOG_DIR / "server.log")

# Dump raw access noise here
accesslog = str(IGNORE_DIR / "access.log")
access_log_format = '%(h)s %(l)s %(u)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

import os

workers = 4
bind = "0.0.0.0:10000"
timeout = 300
worker_class = "gevent"
worker_connections = 1000
keepalive = 2 
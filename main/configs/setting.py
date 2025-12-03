import os

# Robot server target. Override via environment variables when deploying across networks (e.g., ZeroTier).
# Example: ROBOT_HOST=10.147.0.23 ROBOT_PORT=4000 python -m webapp.app
HOST = os.getenv('ROBOT_HOST', '192.168.0.154')
try:
    PORT = int(os.getenv('ROBOT_PORT', '4000'))
except Exception:
    PORT = 4000

import http.server
import logging
import os
import socket
import socketserver
import threading
import time

from .path_utils import resolve_path

logger = logging.getLogger(__name__)

HTTP_SERVER = None
HTTP_PORT = None
DEFAULT_PORT = 57623
NETWORK_DIR = "pyvis_files"


def is_port_available(port: int) -> bool:
	"""Check if a port is available for binding"""
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(("0.0.0.0", port))
			return True
	except OSError:
		return False


def cleanup_old_files(directory: str):
	"""Clean up old pyvis html files"""
	try:
		if os.path.exists(directory):
			current_time = time.time()
			for filename in os.listdir(directory):
				filepath = os.path.join(directory, filename)
				# Remove files older than 30 minutes
				if os.path.getmtime(filepath) < current_time - 1800:
					os.remove(filepath)
	except Exception as e:
		logger.error(f"Cleanup error: {e}")


def find_free_port():
	"""Find a free port for the HTTP server"""

	if is_port_available(DEFAULT_PORT):
		return DEFAULT_PORT

	for port in range(DEFAULT_PORT + 1, DEFAULT_PORT + 5):
		if is_port_available(port):
			return port

	# Fall back to random ports
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("0.0.0.0", 0))
		s.listen(1)
		port = s.getsockname()[1]
	return port


def get_current_port():
	"""Get the current HTTP server port"""
	global HTTP_PORT
	if HTTP_PORT is None:
		HTTP_PORT = find_free_port()
	return HTTP_PORT


def setup_http_server():
	"""Setup a simple HTTP server to serve network files"""
	global HTTP_SERVER, HTTP_PORT

	network_dir_path = os.path.join(os.getcwd(), "ctinexus_output")
	if not os.path.exists(network_dir_path):
		os.makedirs(network_dir_path)

	cleanup_old_files(network_dir_path)

	# Find free port
	HTTP_PORT = get_current_port()

	# Create HTTP server
	class NetworkHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
		def translate_path(self, path):
			if path.startswith("/lib/bindings/"):
				return os.path.join(resolve_path("lib/bindings"), path[len("/lib/bindings/") :])
			return super().translate_path(path)

		def __init__(self, *args, **kwargs):
			super().__init__(*args, directory=network_dir_path, **kwargs)

		def end_headers(self):
			# Add CORS headers to allow iframe embedding
			self.send_header("Access-Control-Allow-Origin", "*")
			self.send_header("Access-Control-Allow-Methods", "GET")
			self.send_header("Access-Control-Allow-Headers", "*")
			super().end_headers()

	# Start server in background thread
	def start_server():
		global HTTP_SERVER, HTTP_PORT
		with socketserver.TCPServer(("0.0.0.0", HTTP_PORT), NetworkHTTPRequestHandler) as httpd:
			HTTP_SERVER = httpd
			HTTP_PORT = httpd.server_address[1]
			logger.debug(f"Network file server running on http://0.0.0.0:{HTTP_PORT}")
			httpd.serve_forever()

	server_thread = threading.Thread(target=start_server, daemon=True)
	server_thread.start()
	time.sleep(0.5)

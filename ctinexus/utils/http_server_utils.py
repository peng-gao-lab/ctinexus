import http.server
import logging
import os
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request

from .path_utils import resolve_path

logger = logging.getLogger(__name__)

HTTP_SERVER = None
HTTP_PORT = None
SERVER_THREAD = None
SERVER_LOCK = threading.Lock()
SERVER_READY = threading.Event()
DEFAULT_PORT = 57623
NETWORK_DIR = "pyvis_files"


class ReusableTCPServer(socketserver.TCPServer):
	"""TCP Server with SO_REUSEADDR enabled to avoid 'Address already in use' errors"""

	allow_reuse_address = True

	def server_close(self):
		"""Override to ensure clean shutdown"""
		try:
			self.shutdown()
		except Exception:
			pass
		super().server_close()


def is_port_in_use(port: int) -> bool:
	"""Check if a port is currently in use"""
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.settimeout(1)
			result = s.connect_ex(("127.0.0.1", port))
			return result == 0
	except Exception:
		return False


def is_port_available(port: int) -> bool:
	"""Check if a port is available for binding"""
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			s.bind(("127.0.0.1", port))
			return True
	except OSError:
		return False


def is_our_server_responding(port: int, timeout: float = 1.0) -> bool:
	"""Check if our HTTP server is responding on the given port with the right content"""
	try:
		url = f"http://127.0.0.1:{port}/"
		req = urllib.request.Request(url, method="GET")
		with urllib.request.urlopen(req, timeout=timeout) as response:
			# Check if it's serving from our ctinexus_output directory
			content = response.read().decode("utf-8", errors="ignore")
			# Our server returns directory listing with network_ files or the index
			return response.status == 200
	except (urllib.error.URLError, socket.timeout, OSError, Exception):
		return False


def cleanup_old_files(directory: str):
	"""Clean up old pyvis html files"""
	try:
		if os.path.exists(directory):
			current_time = time.time()
			for filename in os.listdir(directory):
				filepath = os.path.join(directory, filename)
				# Remove files older than 30 minutes
				if os.path.isfile(filepath) and os.path.getmtime(filepath) < current_time - 1800:
					try:
						os.remove(filepath)
					except OSError:
						pass
	except Exception as e:
		logger.error(f"Cleanup error: {e}")


def find_free_port() -> int:
	"""Find a free port for the HTTP server"""
	# First check if default port has our server already running
	if is_our_server_responding(DEFAULT_PORT):
		logger.debug(f"Found existing CTINexus HTTP server on port {DEFAULT_PORT}")
		return DEFAULT_PORT

	# Check if default port is available
	if is_port_available(DEFAULT_PORT):
		return DEFAULT_PORT

	# Try a range of ports - first check for existing servers
	for port in range(DEFAULT_PORT + 1, DEFAULT_PORT + 20):
		if is_our_server_responding(port):
			logger.debug(f"Found existing CTINexus HTTP server on port {port}")
			return port

	# No existing server found, find an available port
	for port in range(DEFAULT_PORT, DEFAULT_PORT + 100):
		if is_port_available(port):
			return port

	# Fall back to OS-assigned port
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(("127.0.0.1", 0))
		s.listen(1)
		port = s.getsockname()[1]
	return port


def get_current_port() -> int:
	"""Get the current HTTP server port, ensuring server is running"""
	global HTTP_PORT

	with SERVER_LOCK:
		# If we have a port and server is responding, return it
		if HTTP_PORT is not None and is_our_server_responding(HTTP_PORT):
			return HTTP_PORT

		# Check if there's an existing server from this or another process
		for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
			if is_our_server_responding(port):
				HTTP_PORT = port
				logger.debug(f"Reusing existing HTTP server on port {port}")
				return HTTP_PORT

		# No existing server found, start our own
		_ensure_server_running()
		return HTTP_PORT


def _shutdown_server():
	"""Shutdown the current HTTP server"""
	global HTTP_SERVER, HTTP_PORT, SERVER_THREAD

	SERVER_READY.clear()

	if HTTP_SERVER is not None:
		try:
			HTTP_SERVER.shutdown()
		except Exception:
			pass
		try:
			HTTP_SERVER.server_close()
		except Exception:
			pass
		HTTP_SERVER = None

	HTTP_PORT = None
	SERVER_THREAD = None


def _ensure_server_running():
	"""Ensure the HTTP server is running, start it if not"""
	global HTTP_SERVER, HTTP_PORT, SERVER_THREAD

	# If server thread is alive and responding, nothing to do
	if SERVER_THREAD is not None and SERVER_THREAD.is_alive():
		if HTTP_PORT and is_our_server_responding(HTTP_PORT):
			return

	# Check for existing server from another process first
	for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
		if is_our_server_responding(port):
			HTTP_PORT = port
			logger.debug(f"Found existing HTTP server on port {port}, reusing it")
			SERVER_READY.set()
			return

	# Shutdown any existing server we own
	_shutdown_server()

	# Find a free port
	HTTP_PORT = find_free_port()

	# If that port already has a responding server, use it
	if is_our_server_responding(HTTP_PORT):
		logger.debug(f"Port {HTTP_PORT} already has a responding server")
		SERVER_READY.set()
		return

	network_dir_path = os.path.join(os.getcwd(), "ctinexus_output")
	if not os.path.exists(network_dir_path):
		os.makedirs(network_dir_path)

	cleanup_old_files(network_dir_path)

	# Create HTTP server with custom handler
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
			self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
			self.send_header("Access-Control-Allow-Headers", "*")
			self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
			self.send_header("X-CTINexus-Server", "true")  # Marker to identify our server
			super().end_headers()

		def log_message(self, format, *args):
			# Suppress request logging to avoid cluttering output
			pass

	# Start server in background thread
	server_started = threading.Event()
	server_error = [None]  # Use list to allow modification in nested function

	def start_server():
		global HTTP_SERVER
		try:
			httpd = ReusableTCPServer(("0.0.0.0", HTTP_PORT), NetworkHTTPRequestHandler)
			HTTP_SERVER = httpd
			logger.debug(f"Network file server started on http://0.0.0.0:{HTTP_PORT}")
			server_started.set()
			SERVER_READY.set()
			httpd.serve_forever()
		except Exception as e:
			server_error[0] = e
			server_started.set()
			logger.error(f"Failed to start HTTP server on port {HTTP_PORT}: {e}")

	SERVER_THREAD = threading.Thread(target=start_server, daemon=True, name="HTTPFileServer")
	SERVER_THREAD.start()

	# Wait for server to start (with timeout)
	if not server_started.wait(timeout=5.0):
		logger.error("HTTP server failed to start within timeout")
		return

	if server_error[0] is not None:
		logger.error(f"HTTP server error: {server_error[0]}")
		# Try with a different port
		HTTP_PORT = None
		for port in range(DEFAULT_PORT + 1, DEFAULT_PORT + 100):
			if is_port_available(port):
				HTTP_PORT = port
				break
		if HTTP_PORT:
			return _ensure_server_running()
		return

	# Verify server is responding
	for _ in range(20):
		if is_our_server_responding(HTTP_PORT):
			logger.debug(f"HTTP server verified running on port {HTTP_PORT}")
			return
		time.sleep(0.1)

	logger.warning(f"HTTP server started but not responding on port {HTTP_PORT}")


def setup_http_server():
	"""Setup a simple HTTP server to serve network files"""
	with SERVER_LOCK:
		_ensure_server_running()

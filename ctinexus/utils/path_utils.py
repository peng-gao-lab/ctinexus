import os

# Set BASE_DIR to point to the app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(*relative_path):
	"""
	Resolve paths relative to the current module's location.

	Args:
	    *relative_path: Path components to join

	Returns:
	    str: Absolute path relative to this module's location
	"""
	return os.path.join(BASE_DIR, *relative_path)

import re
import sys
from pathlib import Path

API_KEY_PATTERNS = [
    r'sk-[a-zA-Z0-9_-]{20,50}',

    r'AIzaSy[a-zA-Z0-9_-]{33}',
    r'[a-zA-Z0-9]{30,35}',

    r'["\']([a-zA-Z0-9_-]{32,})["\']',
]

WHITELIST_FILES = [
    '.env.example',
    'test_api_keys.py',
]

EXCLUDE_EXTENSIONS = ['.lock', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.xml']

EXCLUDE_DIRS = ['.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__']

def is_excluded(file_path):
    """Check if file should be excluded from scanning."""
    path = Path(file_path)

    if path.suffix in EXCLUDE_EXTENSIONS:
        return True

    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True

    return False

def check_file(file_path):
    """Check a file for API keys."""
    if is_excluded(file_path):
        return True

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return True

    for pattern in API_KEY_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            print(f"Potential API key found in {file_path}:")
            for match in matches:
                print(f"  - {match}")
            return False

    return True

def main():
    files = sys.argv[1:]
    exit_code = 0

    for file_path in files:
        if not check_file(file_path):
            exit_code = 1

    if exit_code == 1:
        print("\nCommit aborted: Found potential API keys in the files above.")
        print("Please remove these keys before committing.")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()

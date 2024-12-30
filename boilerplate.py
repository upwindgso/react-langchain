from dotenv import load_dotenv
from pathlib import Path
import sys

"""
Boilerplate for preferring to load the dotenv file from the parent/one level up
Failing that, from the usual current directory.
Failing that erroring out.

Pref to load from parent : can be shared across multiple repos / less risk of mising in gitignore
"""

def load_env_file():
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    # Check parent directory first
    parent_env = parent_dir / '.env'
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env)
        print(f"Loaded .env from: {parent_env}")
        return
    
    # Check current directory if parent .env not found
    current_env = current_dir / '.env'
    if current_env.exists():
        load_dotenv(dotenv_path=current_env)
        print(f"Loaded .env from: {current_env}")
        return
    
    # No .env found in either location
    print(f"Error: No .env file found in:")
    print(f"  - {parent_env}")
    print(f"  - {current_env}")
    sys.exit(1)

if __name__ == "__main__":
    load_env_file()

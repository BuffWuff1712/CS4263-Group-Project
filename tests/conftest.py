import sys
from pathlib import Path

# Let tests import the project modules when run with plain `pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

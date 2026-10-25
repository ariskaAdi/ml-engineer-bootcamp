# Membuat `import spam` (ml/src) dan `import app` (services/inference) bisa dipakai di tes capstone.
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "ml" / "src"))
sys.path.insert(0, str(HERE / "services" / "inference"))

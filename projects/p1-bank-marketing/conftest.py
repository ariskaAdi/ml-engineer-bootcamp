# Membuat `import bank_marketing` bisa dipakai di tests tanpa install paket.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

"""Real 100 x 100 m crops around known boulders and boulder-free negatives,
listed in boulders.json."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "boulders.json").read_text(encoding="utf-8"))
BOULDERS = MANIFEST["boulders"]
NEGATIVES = MANIFEST["negatives"]  # squares without boulders, to the author's knowledge
_REF = BOULDERS[0]
assert _REF["id"] == "b01_reference"
REFERENCE_CROP = FIXTURES / str(_REF["crops"][0]["file"])
REFERENCE_BOULDER = (float(_REF["e"]), float(_REF["n"]))

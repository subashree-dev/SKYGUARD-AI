from pathlib import Path


source = Path("src/dashboard.py")

lines = source.read_text(encoding="utf-8").splitlines(True)


sections = {
    "overview.py": (126, 403),
    "sensor_monitoring.py": (403, 653),
    "anomaly_detection.py": (653, 914),
    "analytics.py": (914, 1182),
    "ai_model.py": (1182, 1495),
    "system.py": (1495, len(lines)),
}


header = """import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *

"""


for filename, (start, end) in sections.items():

    # Python list indexes are zero-based.
    section = lines[start:end]

    # Remove the first condition line.
    section = section[1:]

    # Remove the 4-space indentation belonging to the old if/elif block.
    fixed = []

    for line in section:

        if line.startswith("    "):
            fixed.append(line[4:])
        else:
            fixed.append(line)

    output = Path("pages") / filename

    output.write_text(
        header + "".join(fixed),
        encoding="utf-8"
    )

    print(f"Created: {output}")
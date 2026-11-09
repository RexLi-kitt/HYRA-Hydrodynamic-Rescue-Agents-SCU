"""Check published comparison aggregation without running models."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

rows = load('comparison.json')
cases = load('case-results.json')['cases']
with (ROOT / 'comparison.csv').open(encoding='utf-8-sig', newline='') as handle:
    csv_rows = list(csv.DictReader(handle))
assert len(rows) == len(csv_rows)
for row, csv_row in zip(rows, csv_rows):
    subset = [c for c in cases if all(c[k] == row[k] for k in ('model', 'version', 'panel'))]
    assert len(subset) == row['episodes']
    for field in ('total', 'delivered'):
        assert sum(c['metrics'][field] for c in subset) == row[field] == int(csv_row[field])
    assert abs(row['RCR'] - row['delivered'] / row['total']) < 1e-12
    for case in subset:
        assert 0 <= case['metrics']['delivered'] <= case['metrics']['total'] == 64
print(f'PASS: {len(cases)} cases, {len(rows)} aggregate rows, CSV agreement and RCR arithmetic.')

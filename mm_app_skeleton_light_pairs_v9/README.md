# MM Analytics (Tkinter skeleton)

## Run
```bash
python main.py
```

## What you get (MVP)
- Light theme tuned to reduce eye strain
- Sidebar navigation (Overview / Pairs)
- Topbar with date range + L1 time window + buttons
- Background thread for Load / Pairing to keep UI responsive
- Uses the existing modules in `core/`:
  - `fake_mm_logs.py` (data generator)
  - `clean_module.py` (clean + keys)
  - `detect_pairs.py` (L1/L2/L3)


## Pairs view upgrades
- Grouped blocks with expand/collapse
- Click headers to sort within groups
- Buy vs Sell styling
- Filter bar (Level / CP / UND / min |PnL|)

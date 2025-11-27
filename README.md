# Bot Detection Heatmap

Bot score represents the percentage of a site's clicks coming from user agents classified as bots based on volume and conversion thresholds.

## Quick Start

```bash
python3 bot_heatmap.py clicks.xlsx
```

## Usage

```bash
# Basic usage (default: 300 min clicks, 1% max conversion)
python3 bot_heatmap.py clicks.xlsx

# Custom output filename
python3 bot_heatmap.py clicks.xlsx -o report_2025-01-15.png

# Custom bot detection thresholds
python3 bot_heatmap.py clicks.xlsx --min-clicks 500 --max-conversion 0.5

# View all options
python3 bot_heatmap.py --help
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-clicks` | 300 | Minimum clicks to consider for bot detection |
| `--max-conversion` | 1.0 | Maximum conversion rate (%) to classify as bot |
| `-o`, `--output` | bot_heatmap.png | Output file path for heatmap |

## Input Format

Excel file (.xlsx) with columns:
- Site/Market
- User Agent
- Entry Clicks
- Orders

## Output

- PNG heatmap image (grid layout, sorted by bot score)
- Console summary with bot statistics and top offenders

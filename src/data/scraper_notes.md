# Splice.com Scraper Notes

## Overview
This document contains notes and instructions for scraping audio samples from Splice.com using the browser extension.

## Browser Extension Location
The browser extension is located in the old structure at:
- `scraper/browser_extension/`

## Installation
See `scraper/browser_extension/README.md` for installation instructions.

## Data Collection Process

### Step 1: Navigate to Splice.com
1. Go to https://splice.com
2. Browse to the category you want to scrape (Drums or Keys)

### Step 2: Use Extension
1. Click the extension icon
2. Select category (drums or keys)
3. Set number of samples (20 per category)
4. Click "Start Scraping"

### Step 3: Save Data
1. Extension will download metadata as JSON
2. Save to `data/metadata/` directory
3. Manually download audio files to appropriate category folder:
   - `data/raw/drums/` for drum samples
   - `data/raw/keys/` for key samples

## Naming Convention
Files should be named as:
- Drums: `drum_01.wav`, `drum_02.wav`, ..., `drum_20.wav`
- Keys: `key_01.wav`, `key_02.wav`, ..., `key_20.wav`

## Metadata Format
The `samples.csv` file should contain:
- `id`: Unique identifier
- `filename`: Audio filename
- `category`: drums or keys
- `title`: Sample title
- `description`: Sample description
- `tags`: Comma-separated tags
- `bpm`: Tempo (if available)
- `key`: Musical key (if available)

## Important Notes
- Respect Splice.com's terms of service
- Add delays between requests to avoid rate limiting
- The extension extracts metadata; audio downloads may need to be manual
- Verify all 40 samples (20 drums + 20 keys) are collected before proceeding

## Next Steps After Collection
1. Verify all audio files are in correct directories
2. Create/update `samples.csv` with metadata
3. Run preprocessing scripts
4. Generate embeddings

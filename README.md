# WhereIsMyThesis

**Work in progress.** The project is not finished and for the moment I'm not actively working on it, so some parts are still rough (see the known issues at the end).

I was really tired of manually searching for thesis/student positions across Europe, going through department and group pages one by one, so I started writing a scraper which does the work for me. You give it a list of pages, it downloads them, an LLM reads the text and extracts the thesis proposals (title, supervisor, contact, short description and topic), and everything ends up in a small database and in a web page where you can search and filter the results.

For the moment the sources are Master thesis pages in physics, mostly about quantum computing and related fields, and many of them are from UniMi since it's where I study. The idea is to extend it beyond physics at some point.

## How it works

1) Every page listed in `sources.yaml` is downloaded with requests. If the text is too short (pages rendered with JavaScript) it falls back to a headless browser with Playwright.
2) The text is cleaned from menus, footers etc. and sent to the LLM, which returns the proposals as JSON. The output is validated with pydantic before saving it.
3) The results are saved in SQLite (`proposals.db`) and exported to `proposals.json`, which is the file read by `index.html`. Pages already scraped in the last 7 days are skipped.

There are two versions of the scraper:

- `scraper.py` uses Claude (Haiku) through the Anthropic API, so you need an API key.
- `scraper_hybrid.py` first tries a simple regex based extraction and then falls back to a local model with Ollama, so it runs for free on your machine. It also keeps track of the pages that don't exist anymore and removes them from `sources.yaml`.

The code they have in common (fetching, prompt, parsing, database) is in `common.py`.

## Usage

1) Install the requirements:
```
pip install -r requirements.txt
playwright install chromium
```
2) Run one of the two scrapers:
```
export ANTHROPIC_API_KEY=...      # Claude version
python scraper.py

ollama run mistral                # local version, in another terminal
python scraper_hybrid.py
```
Useful options: `--force` to scrape again everything, `--url <URL>` to scrape just one page, `-v` for more detailed logs.

3) Open the web page. Browsers block local files, so serve the folder with `python -m http.server` and go to `localhost:8000`. If `proposals.json` is not there yet the page shows some made-up demo entries.

To add a new source simply add its url (and optionally a name) to `sources.yaml`.

## Known issues / to do

- The regex extraction in `scraper_hybrid.py` is way too naive: it takes random capitalized words as supervisor names, so its results are not reliable yet and it needs to be rewritten.
- The quality of the extraction depends a lot on how the page is written, so for now the results should be checked by hand.
- The data are scraped from public pages and can be outdated, always check the original source before contacting someone.

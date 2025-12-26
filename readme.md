# git-loc

Generate a **per-language lines of code time series** for a Git repository **without checking out** each commit.

This tool walks the linear history of a branch/ref, computes a tree diff at each commit, and updates running totals using **git blob contents** + **tokei** for language detection and counting.

- Uses **tokei** for language detection and for `code / comments / blanks` counts (including embedded-language blobs).
- Avoids expensive checkouts: reads blobs directly from the Git object database.
- Caches per-blob tokei results in-memory, so identical blobs are only counted once per run.
- Optional SVG plot of the top languages over time.

> Note: “lines of code” usually refers to the `code` column (non-blank, non-comment lines). The CSV also includes `comments`, `blanks`, and `lines = code+comments+blanks`.

## Install

### From crates.io

```bash
cargo install git-loc
```

### From source

```bash
git clone https://github.com/banteg/git-loc.git
cd git-loc
cargo install --path .
```

## Usage

### Emit CSV (stdout)

```bash
git-loc --repo .
```

### Emit CSV + SVG plot

```bash
git-loc --repo . --out loc.csv --plot loc.svg
```

### Plot options

```bash
# Use code/comments/blanks/lines for the plot metric
git-loc --repo . --out loc.csv --plot loc.svg --plot-metric code

# Plot top 5 languages
git-loc --repo . --out loc.csv --plot loc.svg --plot-top 5

# Only include selected languages (repeatable or comma-separated)
git-loc --repo . --out loc.csv --plot loc.svg --only Python --only "Jinja2"
git-loc --repo . --out loc.csv --plot loc.svg --only Python,Jinja2
```

### Only count a subdirectory

```bash
git-loc --repo . --subdir src --out loc.csv --plot loc.svg
```

### Disable the progress bar

```bash
git-loc --repo . --no-progress
```

## Output format

The CSV is long-form: one row per `(commit, language)`.

Columns:

- `commit`: commit SHA
- `timestamp`: commit time (unix seconds)
- `datetime`: commit time
- `language`: language name (as reported by tokei)
- `code`: code lines
- `comments`: comment lines
- `blanks`: blank lines
- `lines`: `code + comments + blanks`

This long format is convenient for plotting.

## Plot

`--plot <path.svg>` writes an SVG chart of the top languages ranked by totals at the final commit.

Plot-related flags:

- `--plot-metric code|lines|comments|blanks` (default: `code`)
- `--plot-top N` (default: `8`)
- `--only <LANG>` repeatable or comma-separated
- `--no-progress` disables the progress bar on stderr

## How it works

1. Resolve `--rev` (default `HEAD`).
2. Build the first-parent chain from root → tip.
3. For each commit:
   - Diff `parent_tree → commit_tree`
   - For each changed file:
     - If the old blob exists: subtract its cached per-language tokei counts
     - If the new blob exists: add its cached per-language tokei counts
4. Emit the current per-language totals (and optionally add a plot point).

Counting is done by writing blob bytes to a temporary file (with the original basename so tokei’s path-based detection works), then calling tokei as a library.

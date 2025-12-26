use anyhow::{Context, Result};
use chrono::{FixedOffset, NaiveDateTime, TimeZone};
use clap::{Parser, ValueEnum};
use git2::{DiffOptions, FileMode, Oid, Repository, Sort};
use std::{
    collections::HashMap,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use tempfile::TempDir;
use tokei::{CodeStats, Config as TokeiConfig, LanguageType, Languages};

#[derive(Parser, Debug)]
#[command(name = "git-tokei-timeseries")]
#[command(about = "Per-language SLOC time series using git blobs + tokei", long_about = None)]
struct Args {
    /// Path to the git repo (default: .)
    #[arg(long, default_value = ".")]
    repo: PathBuf,

    /// Rev to walk (branch name, tag, SHA, or any revspec). Default: HEAD
    #[arg(long, default_value = "HEAD")]
    rev: String,

    /// Walk mode:
    /// - all: all commits reachable from rev (uses first parent as diff base per commit)
    /// - first-parent: only the first-parent chain (linear history)
    #[arg(long, value_enum, default_value_t = Mode::All)]
    mode: Mode,

    /// Output CSV path, or "-" for stdout
    #[arg(long, default_value = "-")]
    out: String,

    /// Max blob size (bytes) to feed into tokei (guardrail). Default: 50MB
    #[arg(long, default_value_t = 50 * 1024 * 1024)]
    max_bytes: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum Mode {
    All,
    FirstParent,
}

#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    code: i64,
    comments: i64,
    blanks: i64,
}

impl Counts {
    fn is_zero(&self) -> bool {
        self.code == 0 && self.comments == 0 && self.blanks == 0
    }
    fn lines(&self) -> i64 {
        self.code + self.comments + self.blanks
    }
}

type LangMap = HashMap<LanguageType, Counts>;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BlobKey {
    oid: Oid,
    filename: String, // basename only (what tokei uses for detection)
}

fn is_countable_mode(mode: FileMode) -> bool {
    matches!(
        mode,
        FileMode::Blob | FileMode::BlobExecutable | FileMode::BlobGroupWritable
    )
}

fn format_git_time(t: git2::Time) -> String {
    let secs = t.seconds();
    let offset_seconds = t.offset_minutes() * 60;
    let offset = FixedOffset::east_opt(offset_seconds).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());

    let naive = NaiveDateTime::from_timestamp_opt(secs, 0)
        .unwrap_or_else(|| NaiveDateTime::from_timestamp_opt(0, 0).unwrap());

    offset.from_utc_datetime(&naive).to_rfc3339()
}

fn add_to_map(map: &mut LangMap, lang: LanguageType, code: usize, comments: usize, blanks: usize) {
    let e = map.entry(lang).or_default();
    e.code += code as i64;
    e.comments += comments as i64;
    e.blanks += blanks as i64;
}

fn add_codestats_recursive(map: &mut LangMap, lang: LanguageType, stats: &CodeStats) {
    add_to_map(map, lang, stats.code, stats.comments, stats.blanks);
    for (child_lang, child_stats) in &stats.blobs {
        add_codestats_recursive(map, *child_lang, child_stats);
    }
}

fn map_add(dst: &mut LangMap, src: &LangMap) {
    for (lang, c) in src {
        let e = dst.entry(*lang).or_default();
        e.code += c.code;
        e.comments += c.comments;
        e.blanks += c.blanks;
    }
}

fn map_sub(dst: &mut LangMap, src: &LangMap) {
    for (lang, c) in src {
        let e = dst.entry(*lang).or_default();
        e.code -= c.code;
        e.comments -= c.comments;
        e.blanks -= c.blanks;
    }
}

fn blob_lang_counts(
    repo: &Repository,
    tmpdir: &Path,
    tokei_cfg: &TokeiConfig,
    cache: &mut HashMap<BlobKey, Arc<LangMap>>,
    oid: Oid,
    filename: &str,
    max_bytes: usize,
) -> Result<Arc<LangMap>> {
    let key = BlobKey {
        oid,
        filename: filename.to_string(),
    };

    if let Some(v) = cache.get(&key) {
        return Ok(Arc::clone(v));
    }

    let blob = repo
        .find_blob(oid)
        .with_context(|| format!("find_blob({oid})"))?;

    // Skip binary or huge blobs (guardrail).
    if blob.is_binary() || blob.size() > max_bytes {
        let empty = Arc::new(LangMap::new());
        cache.insert(key, Arc::clone(&empty));
        return Ok(empty);
    }

    // Write blob bytes to a temp file whose name matches the original basename.
    // (tokei uses the filename/extension and may inspect content for shebang, etc.)
    let tmp_path = tmpdir.join(filename);
    std::fs::write(&tmp_path, blob.content())
        .with_context(|| format!("write temp file {:?}", tmp_path))?;

    let mut languages = Languages::new();
    // Signature may be () or Result<...>; using let _ avoids must_use warnings either way.
    let _ = languages.get_statistics(&[tmp_path.as_path()], &[], tokei_cfg);

    // Ensure totals are computed.
    for (_, lang) in languages.iter_mut() {
        lang.total();
    }

    // Flatten: primary language counts + embedded language blobs (recursively).
    let mut totals: LangMap = LangMap::new();
    for (lang_type, lang) in languages.iter() {
        for report in &lang.reports {
            let stats = &report.stats;
            // Primary stats for the file's language.
            add_to_map(&mut totals, *lang_type, stats.code, stats.comments, stats.blanks);
            // Embedded languages.
            for (child_lang, child_stats) in &stats.blobs {
                add_codestats_recursive(&mut totals, *child_lang, child_stats);
            }
        }
    }

    // Prune zeros.
    totals.retain(|_, c| !c.is_zero());

    let arc = Arc::new(totals);
    cache.insert(key, Arc::clone(&arc));
    Ok(arc)
}

fn apply_tree_diff(
    repo: &Repository,
    tmpdir: &Path,
    tokei_cfg: &TokeiConfig,
    blob_cache: &mut HashMap<BlobKey, Arc<LangMap>>,
    totals: &mut LangMap,
    old_tree: Option<&git2::Tree>,
    new_tree: &git2::Tree,
    max_bytes: usize,
) -> Result<()> {
    let mut opts = DiffOptions::new();
    // You can tune diff options here; defaults are usually fine for tree-to-tree diffs.
    let diff = repo
        .diff_tree_to_tree(old_tree, Some(new_tree), Some(&mut opts))
        .context("diff_tree_to_tree")?;

    for delta in diff.deltas() {
        let oldf = delta.old_file();
        let newf = delta.new_file();

        // Skip binary deltas.
        if oldf.is_binary() || newf.is_binary() {
            continue;
        }

        // Old side: subtract counts if it "exists" and is a normal blob.
        if oldf.exists() && is_countable_mode(oldf.mode()) {
            if let Some(p) = oldf.path() {
                if let Some(name) = p.file_name() {
                    let filename = name.to_string_lossy();
                    let counts = blob_lang_counts(
                        repo,
                        tmpdir,
                        tokei_cfg,
                        blob_cache,
                        oldf.id(),
                        &filename,
                        max_bytes,
                    )?;
                    map_sub(totals, &counts);
                }
            }
        }

        // New side: add counts if it "exists" and is a normal blob.
        if newf.exists() && is_countable_mode(newf.mode()) {
            if let Some(p) = newf.path() {
                if let Some(name) = p.file_name() {
                    let filename = name.to_string_lossy();
                    let counts = blob_lang_counts(
                        repo,
                        tmpdir,
                        tokei_cfg,
                        blob_cache,
                        newf.id(),
                        &filename,
                        max_bytes,
                    )?;
                    map_add(totals, &counts);
                }
            }
        }
    }

    // Prune zero entries after applying the diff.
    totals.retain(|_, c| !c.is_zero());
    Ok(())
}

fn write_commit_rows<W: Write>(
    wtr: &mut csv::Writer<W>,
    commit_oid: Oid,
    t: git2::Time,
    totals: &LangMap,
) -> Result<()> {
    let ts = t.seconds().to_string();
    let dt = format_git_time(t);

    let mut items: Vec<(LanguageType, Counts)> = totals.iter().map(|(k, v)| (*k, *v)).collect();
    items.sort_by_key(|(lang, _)| lang.name().to_string());

    for (lang, c) in items {
        wtr.write_record(&[
            commit_oid.to_string(),
            ts.clone(),
            dt.clone(),
            lang.name().to_string(),
            c.code.to_string(),
            c.comments.to_string(),
            c.blanks.to_string(),
            c.lines().to_string(),
        ])?;
    }
    Ok(())
}

fn run_first_parent<W: Write>(
    repo: &Repository,
    tip: Oid,
    tmpdir: &Path,
    tokei_cfg: &TokeiConfig,
    blob_cache: &mut HashMap<BlobKey, Arc<LangMap>>,
    wtr: &mut csv::Writer<W>,
    max_bytes: usize,
) -> Result<()> {
    // Build the first-parent chain (OIDs), then reverse so we apply diffs root->tip.
    let mut chain: Vec<Oid> = Vec::new();
    let mut cur = repo.find_commit(tip)?;
    loop {
        chain.push(cur.id());
        if cur.parent_count() == 0 {
            break;
        }
        cur = cur.parent(0)?;
    }
    chain.reverse();

    let mut totals: LangMap = LangMap::new();
    let mut prev_tree: Option<git2::Tree> = None;

    for oid in chain {
        let commit = repo.find_commit(oid)?;
        let tree = commit.tree()?;

        apply_tree_diff(
            repo,
            tmpdir,
            tokei_cfg,
            blob_cache,
            &mut totals,
            prev_tree.as_ref(),
            &tree,
            max_bytes,
        )?;

        write_commit_rows(wtr, oid, commit.time(), &totals)?;
        prev_tree = Some(tree);
    }

    Ok(())
}

fn run_all<W: Write>(
    repo: &Repository,
    tip: Oid,
    tmpdir: &Path,
    tokei_cfg: &TokeiConfig,
    blob_cache: &mut HashMap<BlobKey, Arc<LangMap>>,
    wtr: &mut csv::Writer<W>,
    max_bytes: usize,
) -> Result<()> {
    // We want parents-before-children order so each commit's first parent is already processed.
    let mut revwalk = repo.revwalk()?;
    revwalk.push(tip)?;
    revwalk.set_sorting(Sort::TOPOLOGICAL | Sort::REVERSE)?;

    // First pass: collect commit list + first-parent relation + child counts.
    let mut commits: Vec<Oid> = Vec::new();
    let mut first_parent: HashMap<Oid, Option<Oid>> = HashMap::new();
    let mut remaining_children: HashMap<Oid, usize> = HashMap::new();

    for oid_res in revwalk {
        let oid = oid_res?;
        commits.push(oid);

        let c = repo.find_commit(oid)?;
        let p = if c.parent_count() > 0 {
            Some(c.parent_id(0)?)
        } else {
            None
        };
        first_parent.insert(oid, p);

        if let Some(parent_oid) = p {
            *remaining_children.entry(parent_oid).or_insert(0) += 1;
        }
    }

    // Second pass: DP over first-parent edges.
    // Store only commits that still have unprocessed children.
    let mut totals_store: HashMap<Oid, LangMap> = HashMap::new();

    for oid in commits {
        let commit = repo.find_commit(oid)?;
        let tree = commit.tree()?;

        let parent_oid_opt = *first_parent
            .get(&oid)
            .expect("first_parent map should have every commit");

        let mut totals: LangMap = if let Some(parent_oid) = parent_oid_opt {
            totals_store
                .get(&parent_oid)
                .with_context(|| format!("missing totals for parent {parent_oid}"))?
                .clone()
        } else {
            LangMap::new()
        };

        let parent_tree = if let Some(parent_oid) = parent_oid_opt {
            Some(repo.find_commit(parent_oid)?.tree()?)
        } else {
            None
        };

        apply_tree_diff(
            repo,
            tmpdir,
            tokei_cfg,
            blob_cache,
            &mut totals,
            parent_tree.as_ref(),
            &tree,
            max_bytes,
        )?;

        write_commit_rows(wtr, oid, commit.time(), &totals)?;

        // This commit only needs to be stored if it has children.
        let child_count = remaining_children.get(&oid).copied().unwrap_or(0);
        if child_count > 0 {
            totals_store.insert(oid, totals);
        }

        // We just consumed one child of our parent (if any); maybe drop the parent totals.
        if let Some(parent_oid) = parent_oid_opt {
            if let Some(cnt) = remaining_children.get_mut(&parent_oid) {
                *cnt = cnt.saturating_sub(1);
                if *cnt == 0 {
                    totals_store.remove(&parent_oid);
                }
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let repo = Repository::discover(&args.repo)
        .with_context(|| format!("discover repo at {:?}", args.repo))?;

    let obj = repo
        .revparse_single(&args.rev)
        .with_context(|| format!("revparse {}", args.rev))?;
    let tip_commit = obj
        .peel_to_commit()
        .with_context(|| format!("peel {} to commit", args.rev))?;
    let tip = tip_commit.id();

    let tmp = TempDir::new().context("create tempdir")?;
    let tmpdir = tmp.path();

    let tokei_cfg = TokeiConfig::default();
    let mut blob_cache: HashMap<BlobKey, Arc<LangMap>> = HashMap::new();

    // Output writer
    let out: Box<dyn Write> = if args.out == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(
            std::fs::File::create(&args.out)
                .with_context(|| format!("create output file {}", args.out))?,
        )
    };

    let mut wtr = csv::Writer::from_writer(out);
    wtr.write_record(&[
        "commit",
        "timestamp",
        "datetime",
        "language",
        "code",
        "comments",
        "blanks",
        "lines",
    ])?;

    match args.mode {
        Mode::FirstParent => run_first_parent(
            &repo,
            tip,
            tmpdir,
            &tokei_cfg,
            &mut blob_cache,
            &mut wtr,
            args.max_bytes,
        )?,
        Mode::All => run_all(
            &repo,
            tip,
            tmpdir,
            &tokei_cfg,
            &mut blob_cache,
            &mut wtr,
            args.max_bytes,
        )?,
    }

    wtr.flush()?;
    Ok(())
}

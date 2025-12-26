use anyhow::{Context, Result};
use chrono::{DateTime, Duration, FixedOffset, TimeZone, Utc};
use clap::{Parser, ValueEnum};
use git2::{DiffOptions, FileMode, Oid, Repository};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use plotters::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::{Duration as StdDuration, Instant},
};
use tempfile::TempDir;
use tokei::{CodeStats, Config as TokeiConfig, LanguageType, Languages};

#[derive(Parser, Debug)]
#[command(name = "git-loc")]
#[command(
    about = "Per-language SLOC time series using git blobs + tokei (first-parent chain only)",
    long_about = None
)]
struct Args {
    /// Path to the git repo (default: .)
    #[arg(long, default_value = ".")]
    repo: PathBuf,

    /// Rev to walk (branch name, tag, SHA, or any revspec). Default: HEAD
    #[arg(long, default_value = "HEAD")]
    rev: String,

    /// Output CSV path, or "-" for stdout
    #[arg(long, default_value = "-")]
    out: String,

    /// Output SVG plot path (top languages over time)
    #[arg(long)]
    plot: Option<PathBuf>,

    /// Plot metric (code, lines, comments, blanks)
    #[arg(long, value_enum, default_value_t = PlotMetric::Code)]
    plot_metric: PlotMetric,

    /// Plot top N languages
    #[arg(long, default_value_t = 8)]
    plot_top: usize,

    /// Only include selected languages (repeatable or comma-separated)
    #[arg(long, value_delimiter = ',', value_name = "LANG")]
    only: Vec<String>,

    /// Only include files under this subdir (repo-relative, e.g. "src/")
    #[arg(long)]
    subdir: Option<PathBuf>,

    /// Disable the progress bar
    #[arg(long)]
    no_progress: bool,

    /// Max blob size (bytes) to feed into tokei (guardrail). Default: 50MB
    #[arg(long, default_value_t = 50 * 1024 * 1024)]
    max_bytes: usize,
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
    fn metric(&self, metric: PlotMetric) -> i64 {
        match metric {
            PlotMetric::Code => self.code,
            PlotMetric::Lines => self.lines(),
            PlotMetric::Comments => self.comments,
            PlotMetric::Blanks => self.blanks,
        }
    }
}

type LangMap = HashMap<LanguageType, Counts>;

#[derive(ValueEnum, Debug, Clone, Copy)]
enum PlotMetric {
    Code,
    Lines,
    Comments,
    Blanks,
}

impl PlotMetric {
    fn label(self) -> &'static str {
        match self {
            PlotMetric::Code => "code lines",
            PlotMetric::Lines => "lines",
            PlotMetric::Comments => "comment lines",
            PlotMetric::Blanks => "blank lines",
        }
    }
}

#[derive(Debug, Default)]
struct PlotData {
    times: Vec<i64>,
    series: HashMap<LanguageType, Vec<i64>>,
}

impl PlotData {
    fn push_snapshot(
        &mut self,
        ts: i64,
        totals: &LangMap,
        metric: PlotMetric,
        only: Option<&HashSet<LanguageType>>,
    ) {
        let idx = self.times.len();
        self.times.push(ts);

        for (lang, values) in self.series.iter_mut() {
            let value = totals.get(lang).map(|c| c.metric(metric)).unwrap_or(0);
            values.push(value);
        }

        let mut new_langs = Vec::new();
        for lang in totals.keys() {
            if let Some(only_langs) = only {
                if !only_langs.contains(lang) {
                    continue;
                }
            }
            if !self.series.contains_key(lang) {
                new_langs.push(*lang);
            }
        }

        for lang in new_langs {
            let value = totals.get(&lang).map(|c| c.metric(metric)).unwrap_or(0);
            let mut values = vec![0; idx];
            values.push(value);
            self.series.insert(lang, values);
        }
    }
}

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

fn normalize_subdir(subdir: &Path) -> Result<PathBuf> {
    use std::path::Component;

    let mut out = PathBuf::new();
    for component in subdir.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                anyhow::bail!("subdir must not contain '..' components")
            }
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    Ok(out)
}

fn path_allowed(path: &Path, subdir: Option<&Path>) -> bool {
    match subdir {
        Some(prefix) => prefix.as_os_str().is_empty() || path.starts_with(prefix),
        None => true,
    }
}

fn format_git_time(t: git2::Time) -> String {
    let secs = t.seconds();
    let offset_seconds = t.offset_minutes() * 60;
    let offset =
        FixedOffset::east_opt(offset_seconds).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());

    let naive = DateTime::<Utc>::from_timestamp(secs, 0)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap())
        .naive_utc();

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
    languages.get_statistics(&[tmp_path.as_path()], &[], tokei_cfg);

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
            add_to_map(
                &mut totals,
                *lang_type,
                stats.code,
                stats.comments,
                stats.blanks,
            );
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
    subdir: Option<&Path>,
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
                if path_allowed(p, subdir) {
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
        }

        // New side: add counts if it "exists" and is a normal blob.
        if newf.exists() && is_countable_mode(newf.mode()) {
            if let Some(p) = newf.path() {
                if path_allowed(p, subdir) {
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
    only_langs: Option<&HashSet<LanguageType>>,
) -> Result<()> {
    let ts = t.seconds().to_string();
    let dt = format_git_time(t);

    let mut items: Vec<(LanguageType, Counts)> = totals
        .iter()
        .filter(|(lang, _)| only_langs.is_none_or(|set| set.contains(lang)))
        .map(|(k, v)| (*k, *v))
        .collect();
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
    mut plot_data: Option<&mut PlotData>,
    wtr: &mut csv::Writer<W>,
    progress: Option<&ProgressBar>,
    only_langs: Option<&HashSet<LanguageType>>,
    subdir: Option<&Path>,
    plot_metric: PlotMetric,
    max_bytes: usize,
) -> Result<(usize, LangMap)> {
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
    let chain_len = chain.len();
    if let Some(pb) = progress {
        pb.set_length(chain_len as u64);
        pb.set_position(0);
        pb.set_message("processing commits");
    }

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
            subdir,
            max_bytes,
        )?;

        if let Some(buf) = plot_data.as_deref_mut() {
            buf.push_snapshot(commit.time().seconds(), &totals, plot_metric, only_langs);
        }

        write_commit_rows(wtr, oid, commit.time(), &totals, only_langs)?;
        if let Some(pb) = progress {
            pb.inc(1);
        }
        prev_tree = Some(tree);
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    Ok((chain_len, totals))
}

fn write_plot(
    path: &Path,
    title: &str,
    metric: PlotMetric,
    plot_top: usize,
    plot_data: &PlotData,
    final_totals: &LangMap,
) -> Result<()> {
    if plot_data.times.is_empty() {
        return Ok(());
    }

    if path == Path::new("-") {
        anyhow::bail!("plot output must be a file path (SVG), not stdout");
    }

    let mut langs: Vec<(LanguageType, i64)> = final_totals
        .iter()
        .filter(|(lang, _)| plot_data.series.contains_key(*lang))
        .map(|(lang, c)| (*lang, c.metric(metric)))
        .filter(|(_, lines)| *lines > 0)
        .collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.name().cmp(b.0.name())));

    let top_langs: Vec<LanguageType> = langs
        .into_iter()
        .take(plot_top)
        .map(|(lang, _)| lang)
        .collect();
    if top_langs.is_empty() {
        return Ok(());
    }

    let mut ordered: Vec<(i64, usize)> = plot_data
        .times
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, ts)| (ts, idx))
        .collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let start_ts = ordered
        .first()
        .expect("plot data is non-empty by construction")
        .0;
    let end_ts = ordered
        .last()
        .expect("plot data is non-empty by construction")
        .0;

    let start_dt = Utc
        .timestamp_opt(start_ts, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    let mut end_dt = Utc
        .timestamp_opt(end_ts, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    if end_dt <= start_dt {
        end_dt = start_dt + Duration::seconds(1);
    }

    let mut y_max: i64 = 0;
    for lang in &top_langs {
        if let Some(series) = plot_data.series.get(lang) {
            for value in series {
                y_max = y_max.max(*value);
            }
        }
    }
    if y_max <= 0 {
        y_max = 1;
    }

    let root = SVGBackend::new(path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            format!("{title} {} over time", metric.label()),
            ("sans-serif", 26),
        )
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(start_dt..end_dt, 0i64..y_max)?;

    chart
        .configure_mesh()
        .x_labels(8)
        .y_labels(10)
        .x_label_formatter(&|dt| dt.format("%Y-%m-%d").to_string())
        .y_label_formatter(&|v| v.to_string())
        .light_line_style(&WHITE.mix(0.2))
        .draw()?;

    for (idx, lang) in top_langs.iter().enumerate() {
        let color = Palette99::pick(idx).stroke_width(2);
        let mut samples: Vec<(i64, i64)> = Vec::with_capacity(ordered.len());
        let series = plot_data
            .series
            .get(lang)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        for (ts, idx) in &ordered {
            let value = series.get(*idx).copied().unwrap_or(0);
            if let Some((last_ts, last_value)) = samples.last_mut() {
                if *last_ts == *ts {
                    *last_value = value;
                    continue;
                }
            }
            samples.push((*ts, value));
        }

        let mut step_points: Vec<(DateTime<Utc>, i64)> =
            Vec::with_capacity(samples.len().saturating_mul(2));
        if let Some((first_ts, first_value)) = samples.first().copied() {
            let first_dt = Utc
                .timestamp_opt(first_ts, 0)
                .single()
                .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
            step_points.push((first_dt, first_value));
            for window in samples.windows(2) {
                let (prev_ts, prev_value) = window[0];
                let (cur_ts, cur_value) = window[1];
                let cur_dt = Utc
                    .timestamp_opt(cur_ts, 0)
                    .single()
                    .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
                if cur_ts != prev_ts {
                    step_points.push((cur_dt, prev_value));
                }
                step_points.push((cur_dt, cur_value));
            }
        }

        chart
            .draw_series(LineSeries::new(step_points, color))?
            .label(lang.name().to_string())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
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

    let only_langs = if args.only.is_empty() {
        None
    } else {
        let mut set = HashSet::new();
        for raw in &args.only {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            let lang = LanguageType::from_str(name)
                .map_err(|_| anyhow::anyhow!("unknown language \"{name}\""))?;
            set.insert(lang);
        }
        Some(set)
    };

    let subdir = match args.subdir.as_ref() {
        Some(raw) => {
            let normalized = normalize_subdir(raw)?;
            if normalized.as_os_str().is_empty() {
                None
            } else {
                Some(normalized)
            }
        }
        None => None,
    };

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

    let mut plot_data: Option<PlotData> = args.plot.as_ref().map(|_| PlotData::default());

    let progress = if !args.no_progress && io::stderr().is_terminal() {
        let pb = ProgressBar::new(0);
        pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(8));
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len} {elapsed} ETA {eta}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(StdDuration::from_millis(120));
        Some(pb)
    } else {
        None
    };

    let run_start = Instant::now();
    let (commit_count, final_totals) = run_first_parent(
        &repo,
        tip,
        tmpdir,
        &tokei_cfg,
        &mut blob_cache,
        plot_data.as_mut(),
        &mut wtr,
        progress.as_ref(),
        only_langs.as_ref(),
        subdir.as_deref(),
        args.plot_metric,
        args.max_bytes,
    )?;

    wtr.flush()?;
    let run_elapsed = run_start.elapsed();

    let plot_start = Instant::now();
    if let Some(plot_path) = args.plot.as_ref() {
        if let Some(data) = plot_data.as_ref() {
            let repo_dir = repo.workdir().unwrap_or_else(|| repo.path());
            let repo_name = repo_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("git-loc");
            write_plot(
                plot_path,
                repo_name,
                args.plot_metric,
                args.plot_top,
                data,
                &final_totals,
            )?;
        }
    }
    let plot_elapsed = plot_start.elapsed();

    let total_elapsed = run_start.elapsed();
    let run_secs = run_elapsed.as_secs_f64();
    let commits_per_sec = if run_secs > 0.0 {
        commit_count as f64 / run_secs
    } else {
        0.0
    };
    eprintln!(
        "git-loc: processed {commit_count} commits in {:.2}s ({:.2} commits/s); plot {:.2}s; total {:.2}s",
        run_elapsed.as_secs_f64(),
        commits_per_sec,
        plot_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64()
    );

    Ok(())
}

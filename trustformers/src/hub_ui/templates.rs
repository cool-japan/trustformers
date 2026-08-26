//! HTML page and CSS stylesheet generation for the Hub UI's server-rendered views.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::repository::ModelRepository;
use super::state::ThemeConfig;
use super::types::{ModelVersion, VersionComparison};

// HTML generation functions

pub(super) fn generate_home_html(repositories: &[ModelRepository], theme: &ThemeConfig) -> String {
    let repo_list = repositories
        .iter()
        .map(|repo| {
            format!(
                r#"<div class="repository-card">
                <h3><a href="/ui/repository/{}">{}</a></h3>
                <p>{}</p>
                <div class="stats">
                    <span>Versions: {}</span>
                    <span>Downloads: {}</span>
                    <span>Stars: {}</span>
                </div>
            </div>"#,
                repo.model_id,
                repo.metadata.name,
                repo.metadata.description.as_ref().unwrap_or(&"No description".to_string()),
                repo.metadata.stats.version_count,
                repo.metadata.stats.total_downloads,
                repo.metadata.stats.stars
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1>🤖 TrustformeRS Model Hub</h1>
        <nav>
            <a href="/ui/">Home</a>
            <a href="/api/v1/">API</a>
        </nav>
    </header>
    <main>
        <h2>Model Repositories</h2>
        <div class="repositories">
            {repo_list}
        </div>
    </main>
</body>
</html>"#,
        css = generate_css(theme),
        repo_list = repo_list
    )
}

pub(super) fn generate_repository_html(
    repository: &ModelRepository,
    theme: &ThemeConfig,
) -> String {
    let versions = repository.list_versions();
    let version_list = versions
        .iter()
        .map(|version| {
            format!(
                r#"<tr>
                <td><a href="/ui/repository/{}/version/{}">{}</a></td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.2} MB</td>
                <td><span class="status-{}">{:?}</span></td>
            </tr>"#,
                repository.model_id,
                version.version,
                version.version,
                version.name.as_ref().unwrap_or(&version.version),
                format_timestamp(version.created_at),
                version.size_bytes as f64 / 1_000_000.0,
                format!("{:?}", version.status).to_lowercase(),
                version.status
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <div class="repository-header">
            <h2>{}</h2>
            <p>{}</p>
            <div class="repository-stats">
                <span>👤 {}</span>
                <span>📦 {} versions</span>
                <span>⬇️ {} downloads</span>
                <span>⭐ {} stars</span>
            </div>
        </div>

        <section>
            <h3>Versions</h3>
            <table class="versions-table">
                <thead>
                    <tr>
                        <th>Version</th>
                        <th>Name</th>
                        <th>Created</th>
                        <th>Size</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {version_list}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"#,
        repository.metadata.name,
        repository.metadata.name,
        repository
            .metadata
            .description
            .as_ref()
            .unwrap_or(&"No description".to_string()),
        repository.metadata.owner,
        repository.metadata.stats.version_count,
        repository.metadata.stats.total_downloads,
        repository.metadata.stats.stars,
        css = generate_css(theme),
        version_list = version_list
    )
}

pub(super) fn generate_version_html(
    repository: &ModelRepository,
    version: &ModelVersion,
    theme: &ThemeConfig,
) -> String {
    let metrics_html = if let Some(metrics) = &version.metrics {
        format!(
            r#"<div class="metrics">
                <h4>Performance Metrics</h4>
                <div class="metric-grid">
                    {}
                    {}
                    {}
                    {}
                </div>
            </div>"#,
            metrics
                .accuracy
                .map(|a| format!("<div>Accuracy: {:.3}</div>", a))
                .unwrap_or_default(),
            metrics.loss.map(|l| format!("<div>Loss: {:.3}</div>", l)).unwrap_or_default(),
            metrics
                .inference_speed
                .map(|s| format!("<div>Speed: {:.1} tok/s</div>", s))
                .unwrap_or_default(),
            metrics
                .memory_usage
                .map(|m| format!("<div>Memory: {:.1} MB</div>", m))
                .unwrap_or_default(),
        )
    } else {
        "".to_string()
    };
    let changes_html = version
        .changes
        .iter()
        .map(|change| {
            format!(
                r#"<div class="change-item">
                <span class="change-type-{}">{:?}</span>
                <span class="change-path">{}</span>
                <span class="change-size">{:.2} MB</span>
            </div>"#,
                format!("{:?}", change.change_type).to_lowercase(),
                change.change_type,
                change.path,
                change.new_size as f64 / 1_000_000.0
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} {} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <nav class="breadcrumb">
            <a href="/ui/repository/{}">{}</a> / {}
        </nav>

        <div class="version-header">
            <h2>{} {}</h2>
            <span class="status-{}">{:?}</span>
        </div>

        <div class="version-info">
            <p>{}</p>
            <div class="version-meta">
                <span>👤 {}</span>
                <span>📅 {}</span>
                <span>📦 {:.2} MB</span>
                <span>🏷️ {}</span>
            </div>
        </div>

        {metrics_html}

        <div class="changes">
            <h4>Changes</h4>
            <div class="changes-list">
                {changes_html}
            </div>
        </div>

        <div class="actions">
            <button onclick="downloadVersion()">⬇️ Download</button>
            <button onclick="showComparison()">🔍 Compare</button>
        </div>
    </main>

    <script>
        function downloadVersion() {{
            fetch('/api/v1/repositories/{}/download/{}', {{method: 'GET'}})
                .then(response => response.json())
                .then(data => alert('Download started: ' + data.status));
        }}

        function showComparison() {{
            // Implementation for version comparison UI
            alert('Version comparison feature coming soon!');
        }}
    </script>
</body>
</html>"#,
        repository.metadata.name,
        version.version,
        repository.model_id,
        repository.metadata.name,
        version.version,
        repository.metadata.name,
        version.name.as_ref().unwrap_or(&version.version),
        format!("{:?}", version.status).to_lowercase(),
        version.status,
        version.description.as_ref().unwrap_or(&"No description".to_string()),
        version.author.as_ref().unwrap_or(&"Unknown".to_string()),
        format_timestamp(version.created_at),
        version.size_bytes as f64 / 1_000_000.0,
        version.tags.join(", "),
        repository.model_id,
        version.version,
        css = generate_css(theme),
        metrics_html = metrics_html,
        changes_html = changes_html
    )
}

pub(super) fn generate_comparison_html(
    repository: &ModelRepository,
    comparison: &VersionComparison,
    theme: &ThemeConfig,
) -> String {
    let performance_rows = vec![
        (
            "Accuracy",
            comparison.performance_diff.accuracy_diff.map(|d| format!("{:+.3}", d)),
        ),
        (
            "Loss",
            comparison.performance_diff.loss_diff.map(|d| format!("{:+.3}", d)),
        ),
        (
            "Speed",
            comparison.performance_diff.speed_diff.map(|d| format!("{:+.1} tok/s", d)),
        ),
        (
            "Memory",
            comparison.performance_diff.memory_diff.map(|d| format!("{:+.1} MB", d)),
        ),
    ]
    .into_iter()
    .map(|(metric, diff)| {
        format!(
            "<tr><td>{}</td><td>{}</td></tr>",
            metric,
            diff.unwrap_or("N/A".to_string())
        )
    })
    .collect::<Vec<_>>()
    .join("\n");
    let changes_rows = comparison
        .changes
        .iter()
        .map(|change| {
            format!(
                r#"<tr>
                <td><span class="change-type-{}">{:?}</span></td>
                <td>{}</td>
                <td>{:.2} MB</td>
                <td>{}</td>
            </tr>"#,
                format!("{:?}", change.change_type).to_lowercase(),
                change.change_type,
                change.path,
                change.new_size as f64 / 1_000_000.0,
                change.description.as_ref().unwrap_or(&"".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compare {} vs {} - TrustformeRS Model Hub</title>
    <style>
        {css}
    </style>
</head>
<body>
    <header>
        <h1><a href="/ui/">🤖 TrustformeRS Model Hub</a></h1>
    </header>
    <main>
        <nav class="breadcrumb">
            <a href="/ui/repository/{}">{}</a> / Compare
        </nav>

        <div class="comparison-header">
            <h2>🔍 Version Comparison</h2>
            <div class="comparison-versions">
                <span class="version-from">{}</span>
                <span class="arrow">→</span>
                <span class="version-to">{}</span>
            </div>
        </div>

        <div class="comparison-summary">
            <div class="size-diff">
                <h4>Size Change</h4>
                <span class="diff-value">{:+.2} MB</span>
            </div>
        </div>

        <section>
            <h3>Performance Changes</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    {performance_rows}
                </tbody>
            </table>
        </section>

        <section>
            <h3>File Changes</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>File</th>
                        <th>Size</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {changes_rows}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"#,
        comparison.from_version,
        comparison.to_version,
        repository.model_id,
        repository.metadata.name,
        comparison.from_version,
        comparison.to_version,
        comparison.size_diff as f64 / 1_000_000.0,
        css = generate_css(theme),
        performance_rows = performance_rows,
        changes_rows = changes_rows
    )
}

fn generate_css(theme: &ThemeConfig) -> String {
    format!(
        r#"
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: {text_color};
            background: {bg_color};
        }}

        header {{
            background: {primary_color};
            color: white;
            padding: 1rem 2rem;
            border-bottom: 3px solid {secondary_color};
        }}

        header h1 {{
            font-size: 1.5rem;
            display: inline-block;
        }}

        header h1 a {{
            color: white;
            text-decoration: none;
        }}

        nav {{
            float: right;
            margin-top: 0.25rem;
        }}

        nav a {{
            color: white;
            text-decoration: none;
            margin-left: 1rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: background 0.2s;
        }}

        nav a:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .breadcrumb {{
            margin-bottom: 1rem;
            color: {secondary_color};
        }}

        .breadcrumb a {{
            color: {primary_color};
            text-decoration: none;
        }}

        .repository-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: box-shadow 0.2s;
        }}

        .repository-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .repository-card h3 {{
            margin-bottom: 0.5rem;
        }}

        .repository-card h3 a {{
            color: {primary_color};
            text-decoration: none;
        }}

        .stats {{
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: {secondary_color};
        }}

        .versions-table, .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        .versions-table th, .versions-table td,
        .comparison-table th, .comparison-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid {border_color};
        }}

        .versions-table th, .comparison-table th {{
            background: {card_bg};
            font-weight: 600;
        }}

        .status-stable {{ color: #10b981; }}
        .status-development {{ color: #f59e0b; }}
        .status-experimental {{ color: #8b5cf6; }}
        .status-deprecated {{ color: #ef4444; }}
        .status-archived {{ color: {secondary_color}; }}

        .change-type-added {{ color: #10b981; }}
        .change-type-modified {{ color: #f59e0b; }}
        .change-type-deleted {{ color: #ef4444; }}
        .change-type-renamed {{ color: #3b82f6; }}

        .version-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .version-info {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .version-meta {{
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: {secondary_color};
        }}

        .metrics {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .changes {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .actions {{
            display: flex;
            gap: 1rem;
        }}

        button {{
            background: {primary_color};
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }}

        button:hover {{
            background: {primary_color}dd;
        }}

        .comparison-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}

        .comparison-versions {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 1.2rem;
        }}

        .version-from, .version-to {{
            background: {card_bg};
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid {border_color};
        }}

        .arrow {{
            color: {secondary_color};
            font-size: 1.5rem;
        }}

        .comparison-summary {{
            background: {card_bg};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }}

        .diff-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: {primary_color};
        }}
        "#,
        primary_color = theme.primary_color,
        secondary_color = theme.secondary_color,
        text_color = if theme.dark_mode { "#e5e7eb" } else { "#111827" },
        bg_color = if theme.dark_mode { "#111827" } else { "#ffffff" },
        card_bg = if theme.dark_mode { "#1f2937" } else { "#f9fafb" },
        border_color = if theme.dark_mode { "#374151" } else { "#e5e7eb" },
    )
}

/// Render a Unix timestamp (seconds) as a human-readable relative time using
/// `chrono`. The previous implementation subtracted `timestamp` from "now" as
/// plain `u64`s, which underflows (panics in debug builds) for any
/// `timestamp` in the future — a real possibility from clock skew or test
/// fixtures, not just a hypothetical. This handles both directions.
pub(super) fn format_timestamp(timestamp: u64) -> String {
    let Ok(timestamp_i64) = i64::try_from(timestamp) else {
        return "unknown time".to_string();
    };
    let Some(then) = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp_i64, 0) else {
        return "unknown time".to_string();
    };
    let delta = chrono::Utc::now().signed_duration_since(then);
    let secs = delta.num_seconds();
    let (abs_secs, suffix) = if secs >= 0 { (secs, "ago") } else { (-secs, "from now") };
    if abs_secs < 60 {
        "just now".to_string()
    } else if abs_secs < 3600 {
        let mins = abs_secs / 60;
        format!("{mins} minute{} {suffix}", if mins == 1 { "" } else { "s" })
    } else if abs_secs < 86400 {
        let hours = abs_secs / 3600;
        format!("{hours} hour{} {suffix}", if hours == 1 { "" } else { "s" })
    } else {
        let days = abs_secs / 86400;
        format!("{days} day{} {suffix}", if days == 1 { "" } else { "s" })
    }
}

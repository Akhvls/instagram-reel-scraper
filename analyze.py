#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    nltk = None
    SentimentIntensityAnalyzer = Any  # type: ignore[assignment]
    NLTK_IMPORT_ERROR = exc
else:
    NLTK_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sentiment in Instagram reel comments JSON and generate "
            "analysis_results.json plus index.html beside the input file."
        )
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the JSON created by scrape_reel_comments.py.",
    )
    return parser.parse_args()


def load_scrape_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Input file is not valid JSON: {path}") from exc

    if isinstance(payload, list):
        return {}, payload

    if isinstance(payload, dict):
        comments = payload.get("comments")
        if not isinstance(comments, list):
            raise RuntimeError(
                "Expected either a top-level array of comments or an object "
                "with a 'comments' array."
            )
        metadata = {key: value for key, value in payload.items() if key != "comments"}
        return metadata, comments

    raise RuntimeError(
        "Expected either a top-level array of comments or an object "
        "with a 'comments' array."
    )


def ensure_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    if nltk is None:
        raise RuntimeError(
            "NLTK is not installed. Install dependencies first with "
            "'pip install -r requirements.txt'."
        ) from NLTK_IMPORT_ERROR

    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = nltk.download(
            "vader_lexicon",
            download_dir=str(NLTK_DATA_DIR),
            quiet=True,
        )
        if not downloaded:
            raise RuntimeError(
                "Failed to download the NLTK VADER lexicon. "
                "Install dependencies and ensure network access is available."
            )
        return SentimentIntensityAnalyzer()


def coerce_number(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_sentiment(compound_score: float) -> str:
    if compound_score >= POSITIVE_THRESHOLD:
        return "positive"
    if compound_score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def pearson_correlation(
    comments: list[dict[str, Any]],
    field_name: str,
) -> dict[str, float | int | None]:
    pairs: list[tuple[float, float]] = []

    for comment in comments:
        sentiment = coerce_number(comment.get("sentiment_compound"))
        metric = coerce_number(comment.get(field_name))
        if sentiment is None or metric is None:
            continue
        pairs.append((sentiment, metric))

    if len(pairs) < 2:
        return {"coefficient": None, "sample_size": len(pairs)}

    sentiment_mean = fmean(sentiment for sentiment, _ in pairs)
    metric_mean = fmean(metric for _, metric in pairs)

    numerator = sum(
        (sentiment - sentiment_mean) * (metric - metric_mean)
        for sentiment, metric in pairs
    )
    sentiment_variance = sum((sentiment - sentiment_mean) ** 2 for sentiment, _ in pairs)
    metric_variance = sum((metric - metric_mean) ** 2 for _, metric in pairs)

    if sentiment_variance == 0 or metric_variance == 0:
        coefficient = None
    else:
        coefficient = numerator / math.sqrt(sentiment_variance * metric_variance)

    return {
        "coefficient": round_float(coefficient),
        "sample_size": len(pairs),
    }


def average_sentiment(comments: list[dict[str, Any]]) -> float | None:
    scores = [
        float(comment["sentiment_compound"])
        for comment in comments
        if comment.get("sentiment_compound") is not None
    ]
    if not scores:
        return None
    return round_float(fmean(scores))


def comment_snapshot(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "comment_id": comment.get("comment_id"),
        "parent_comment_id": comment.get("parent_comment_id"),
        "username": comment.get("username"),
        "user_id": comment.get("user_id"),
        "is_verified": comment.get("is_verified"),
        "text": comment.get("text"),
        "created_at_epoch": comment.get("created_at_epoch"),
        "created_at_iso": comment.get("created_at_iso"),
        "like_count": comment.get("like_count"),
        "reply_count": comment.get("reply_count"),
        "is_reply": comment.get("is_reply"),
        "permalink": comment.get("permalink"),
        "sentiment_compound": comment.get("sentiment_compound"),
        "sentiment_label": comment.get("sentiment_label"),
    }


def top_ranked_comments(
    comments: list[dict[str, Any]],
    *,
    positive: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    def metric_value(comment: dict[str, Any], field_name: str) -> float:
        value = coerce_number(comment.get(field_name))
        return 0.0 if value is None else value

    if positive:
        ranked = sorted(
            comments,
            key=lambda comment: (
                metric_value(comment, "sentiment_compound"),
                metric_value(comment, "like_count"),
                metric_value(comment, "reply_count"),
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            comments,
            key=lambda comment: (
                metric_value(comment, "sentiment_compound"),
                -metric_value(comment, "like_count"),
                -metric_value(comment, "reply_count"),
            ),
        )

    return [comment_snapshot(comment) for comment in ranked[:limit]]


def analyze_comments(
    raw_comments: list[dict[str, Any]],
    analyzer: SentimentIntensityAnalyzer,
) -> dict[str, Any]:
    analyzed_comments: list[dict[str, Any]] = []
    distribution_counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }

    for raw_comment in raw_comments:
        comment = dict(raw_comment)
        text = comment.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        compound_score = analyzer.polarity_scores(text)["compound"]
        sentiment_label = classify_sentiment(compound_score)

        comment["sentiment_compound"] = round_float(compound_score)
        comment["sentiment_label"] = sentiment_label

        analyzed_comments.append(comment)
        distribution_counts[sentiment_label] += 1

    total_comments = len(analyzed_comments)
    distribution = {}
    for label in ("positive", "neutral", "negative"):
        count = distribution_counts[label]
        percentage = 0.0 if total_comments == 0 else round((count / total_comments) * 100, 2)
        distribution[label] = {
            "count": count,
            "percentage": percentage,
        }

    top_level_comments = [comment for comment in analyzed_comments if not comment.get("is_reply")]
    reply_comments = [comment for comment in analyzed_comments if comment.get("is_reply")]

    return {
        "comments": analyzed_comments,
        "summary": {
            "total_comments": total_comments,
            "sentiment_distribution": distribution,
            "average_sentiment_all_comments": average_sentiment(analyzed_comments),
            "average_sentiment_top_level_comments": average_sentiment(top_level_comments),
            "average_sentiment_replies": average_sentiment(reply_comments),
            "correlations": {
                "sentiment_vs_like_count": pearson_correlation(
                    analyzed_comments,
                    "like_count",
                ),
                "sentiment_vs_reply_count": pearson_correlation(
                    analyzed_comments,
                    "reply_count",
                ),
            },
        },
        "top_comments": {
            "most_positive": top_ranked_comments(analyzed_comments, positive=True),
            "most_negative": top_ranked_comments(analyzed_comments, positive=False),
        },
    }


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_percentage(value: float | None) -> str:
    if value is None:
        return "0.00%"
    return f"{value:.2f}%"


def format_correlation_text(name: str, payload: dict[str, Any]) -> str:
    coefficient = payload.get("coefficient")
    sample_size = payload.get("sample_size")
    coefficient_text = "n/a" if coefficient is None else f"{coefficient:.4f}"
    return f"{name}: {coefficient_text} (n={sample_size})"


def escape_json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_comment_list(title: str, comments: list[dict[str, Any]]) -> str:
    items: list[str] = []

    for comment in comments:
        username = html.escape(comment.get("username") or "unknown")
        text = html.escape(str(comment.get("text") or "").strip() or "[no text]")
        sentiment = comment.get("sentiment_compound")
        like_count = comment.get("like_count")
        reply_count = comment.get("reply_count")
        permalink = comment.get("permalink")
        created_at = comment.get("created_at_iso")
        comment_kind = "reply" if comment.get("is_reply") else "top-level"

        footer_parts = [
            f"score {sentiment:.4f}" if isinstance(sentiment, float) else "score n/a",
            f"{format_number(like_count)} likes",
            f"{format_number(reply_count)} replies",
            comment_kind,
        ]
        if created_at:
            footer_parts.append(html.escape(str(created_at)))
        footer = " | ".join(footer_parts)

        link_html = ""
        if permalink:
            safe_permalink = html.escape(str(permalink), quote=True)
            link_html = f' <a href="{safe_permalink}">permalink</a>'

        items.append(
            "\n".join(
                [
                    '<li class="comment-item">',
                    f'  <div class="comment-meta">@{username}{link_html}</div>',
                    f'  <blockquote>{text}</blockquote>',
                    f'  <div class="comment-footer">{footer}</div>',
                    "</li>",
                ]
            )
        )

    if not items:
        items.append('<li class="comment-item empty">No comments available.</li>')

    return "\n".join(
        [
            '<article class="comment-panel">',
            f"  <h3>{html.escape(title)}</h3>",
            '  <ol class="comment-list">',
            *items,
            "  </ol>",
            "</article>",
        ]
    )


def render_html_report(analysis: dict[str, Any]) -> str:
    scrape_metadata = analysis.get("scrape_metadata") or {}
    summary = analysis["summary"]
    top_comments = analysis["top_comments"]
    distribution = summary["sentiment_distribution"]

    positive_pct = distribution["positive"]["percentage"]
    neutral_pct = distribution["neutral"]["percentage"]
    negative_pct = distribution["negative"]["percentage"]

    scatter_points = [
        {
            "x": comment["sentiment_compound"],
            "y": int(float(comment["like_count"])),
            "username": comment.get("username") or "unknown",
            "text": (comment.get("text") or "").strip() or "[no text]",
        }
        for comment in analysis["comments"]
        if comment.get("sentiment_compound") is not None
        and coerce_number(comment.get("like_count")) is not None
    ]

    metadata_rows = [
        ("Reel URL", analysis.get("reel_url")),
        ("Shortcode", analysis.get("shortcode")),
        ("Media ID", analysis.get("media_id")),
        ("Source JSON", analysis.get("source_file")),
        ("Scraped At", scrape_metadata.get("scraped_at")),
        (
            "Export Counts",
            (
                f"{format_number(scrape_metadata.get('exported_total_count'))} total | "
                f"{format_number(scrape_metadata.get('exported_top_level_comment_count'))} top-level | "
                f"{format_number(scrape_metadata.get('exported_reply_count'))} replies"
            ),
        ),
        (
            "Reported Top-Level Count",
            scrape_metadata.get("reported_top_level_comment_count"),
        ),
        ("Partial Export", scrape_metadata.get("is_partial")),
        ("Analysis Generated", analysis.get("generated_at")),
    ]

    warning_items = []
    for warning in scrape_metadata.get("warnings") or []:
        warning_items.append(f"<li>{html.escape(str(warning))}</li>")

    warnings_html = ""
    if warning_items:
        warnings_html = "\n".join(
            [
                '<section class="panel">',
                "  <h2>Warnings</h2>",
                '  <ul class="warning-list">',
                *warning_items,
                "  </ul>",
                "</section>",
            ]
        )

    scatter_section = ""
    if scatter_points:
        scatter_section = "\n".join(
            [
                '<section class="panel">',
                "  <h2>Sentiment vs Likes</h2>",
                '  <div class="chart-wrap">',
                '    <canvas id="likesChart" aria-label="Scatter plot of sentiment versus likes"></canvas>',
                "  </div>",
                "</section>",
            ]
        )
    else:
        scatter_section = "\n".join(
            [
                '<section class="panel">',
                "  <h2>Sentiment vs Likes</h2>",
                "  <p>No comments with numeric like counts were available for plotting.</p>",
                "</section>",
            ]
        )

    metadata_html = "\n".join(
        [
            '<section class="panel">',
            "  <h2>Scrape Metadata</h2>",
            '  <dl class="metadata-grid">',
            *[
                (
                    f"    <dt>{html.escape(label)}</dt>"
                    f"<dd>{html.escape(format_number(value)) if label != 'Reel URL' or not value else f'<a href=\"{html.escape(str(value), quote=True)}\">{html.escape(str(value))}</a>'}</dd>"
                )
                for label, value in metadata_rows
            ],
            "  </dl>",
            "</section>",
        ]
    )

    correlation_text = [
        format_correlation_text(
            "Pearson r (sentiment, likes)",
            summary["correlations"]["sentiment_vs_like_count"],
        ),
        format_correlation_text(
            "Pearson r (sentiment, replies)",
            summary["correlations"]["sentiment_vs_reply_count"],
        ),
        (
            "Average sentiment, top-level comments: "
            f"{format_number(summary['average_sentiment_top_level_comments'])}"
        ),
        (
            "Average sentiment, replies: "
            f"{format_number(summary['average_sentiment_replies'])}"
        ),
    ]

    correlation_paragraphs = "\n".join(
        f"  <p>{html.escape(line)}</p>" for line in correlation_text
    )

    chart_script = ""
    if scatter_points:
        chart_script = f"""
  <script>
    const scatterPoints = {escape_json_for_script(scatter_points)};
    const ctx = document.getElementById('likesChart');
    new Chart(ctx, {{
      type: 'scatter',
      data: {{
        datasets: [{{
          label: 'Comments',
          data: scatterPoints,
          pointRadius: 4,
          pointHoverRadius: 5,
          pointBackgroundColor: '#36536b',
          pointBorderColor: '#36536b'
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{
            type: 'linear',
            min: -1,
            max: 1,
            title: {{
              display: true,
              text: 'Compound sentiment score'
            }},
            grid: {{
              color: '#d8d2c8'
            }}
          }},
          y: {{
            title: {{
              display: true,
              text: 'Like count'
            }},
            grid: {{
              color: '#d8d2c8'
            }}
          }}
        }},
        plugins: {{
          legend: {{
            display: false
          }},
          tooltip: {{
            callbacks: {{
              label(context) {{
                const raw = context.raw || {{}};
                const username = raw.username || 'unknown';
                const text = raw.text || '[no text]';
                return `@${{username}} | likes=${{raw.y}} | sentiment=${{raw.x.toFixed(4)}} | ${{text}}`;
              }}
            }}
          }}
        }}
      }}
    }});
  </script>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Instagram Reel Comment Sentiment Report</title>
  <style>
    :root {{
      --paper: #f5f1e8;
      --ink: #201a16;
      --muted: #665e57;
      --line: #cfc6b7;
      --positive: #4f7f51;
      --neutral: #989286;
      --negative: #8a4d47;
      --accent: #36536b;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 40px 20px 64px;
      background: var(--paper);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.6;
    }}

    main {{
      max-width: 960px;
      margin: 0 auto;
    }}

    h1, h2 {{
      font-weight: 600;
      letter-spacing: 0.02em;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: 2rem;
    }}

    h2 {{
      margin: 0 0 14px;
      font-size: 1.2rem;
    }}

    p {{
      margin: 0 0 10px;
    }}

    a {{
      color: var(--accent);
    }}

    .lede {{
      color: var(--muted);
      margin-bottom: 24px;
    }}

    .panel {{
      border-top: 1px solid var(--line);
      padding-top: 18px;
      margin-top: 22px;
    }}

    .metadata-grid {{
      display: grid;
      grid-template-columns: minmax(180px, 220px) 1fr;
      gap: 8px 14px;
      margin: 0;
    }}

    .metadata-grid dt {{
      font-weight: 600;
    }}

    .metadata-grid dd {{
      margin: 0;
      color: var(--muted);
    }}

    .distribution-bar {{
      display: flex;
      width: 100%;
      height: 28px;
      border: 1px solid var(--line);
      background: #efe8da;
      overflow: hidden;
    }}

    .segment-positive {{
      background: var(--positive);
      width: {positive_pct:.2f}%;
    }}

    .segment-neutral {{
      background: var(--neutral);
      width: {neutral_pct:.2f}%;
    }}

    .segment-negative {{
      background: var(--negative);
      width: {negative_pct:.2f}%;
    }}

    .distribution-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.98rem;
    }}

    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .legend-swatch {{
      display: inline-block;
      width: 11px;
      height: 11px;
    }}

    .chart-wrap {{
      height: 420px;
      padding-top: 8px;
    }}

    .comment-columns {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 24px;
    }}

    .comment-panel h3 {{
      margin: 0 0 14px;
      font-size: 1.05rem;
      font-weight: 600;
    }}

    .comment-list {{
      margin: 0;
      padding-left: 18px;
    }}

    .comment-item {{
      margin-bottom: 18px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}

    .comment-meta {{
      color: var(--muted);
      font-size: 0.95rem;
    }}

    blockquote {{
      margin: 8px 0 10px;
      padding: 0 0 0 12px;
      border-left: 2px solid var(--line);
      white-space: pre-wrap;
    }}

    .comment-footer {{
      color: var(--muted);
      font-size: 0.92rem;
    }}

    .warning-list {{
      margin: 0;
      padding-left: 18px;
    }}

    .note {{
      color: var(--muted);
      font-size: 0.95rem;
    }}

    @media (max-width: 700px) {{
      body {{
        padding: 28px 16px 48px;
      }}

      .metadata-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <main>
    <header>
      <h1>Instagram Reel Comment Sentiment Report</h1>
      <p class="lede">A minimal static summary of VADER sentiment scores across the scraped reel comment set.</p>
    </header>

{metadata_html}

    <section class="panel">
      <h2>Sentiment Distribution</h2>
      <div class="distribution-bar" aria-label="Sentiment distribution bar">
        <div class="segment-positive"></div>
        <div class="segment-neutral"></div>
        <div class="segment-negative"></div>
      </div>
      <div class="distribution-legend">
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--positive);"></span>Positive: {format_percentage(positive_pct)} ({distribution["positive"]["count"]})</span>
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--neutral);"></span>Neutral: {format_percentage(neutral_pct)} ({distribution["neutral"]["count"]})</span>
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--negative);"></span>Negative: {format_percentage(negative_pct)} ({distribution["negative"]["count"]})</span>
      </div>
      <p class="note">Classification thresholds follow the standard VADER convention: positive ≥ 0.05, neutral between -0.05 and 0.05, negative ≤ -0.05.</p>
    </section>

{scatter_section}

    <section class="panel">
      <h2>Correlation Coefficients</h2>
{correlation_paragraphs}
    </section>

    <section class="panel">
      <h2>Extreme Comments</h2>
      <div class="comment-columns">
{render_comment_list("Top 5 Most Positive Comments", top_comments["most_positive"])}
{render_comment_list("Top 5 Most Negative Comments", top_comments["most_negative"])}
      </div>
    </section>

{warnings_html}
  </main>
{chart_script}
</body>
</html>
"""


def build_analysis_result(
    input_path: Path,
    scrape_metadata: dict[str, Any],
    analyzed_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path.name),
        "reel_url": scrape_metadata.get("reel_url"),
        "shortcode": scrape_metadata.get("shortcode"),
        "media_id": scrape_metadata.get("media_id"),
        "scrape_metadata": scrape_metadata,
        "summary": analyzed_payload["summary"],
        "top_comments": analyzed_payload["top_comments"],
        "comments": analyzed_payload["comments"],
    }


def main() -> int:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()

    scrape_metadata, raw_comments = load_scrape_payload(input_path)
    analyzer = ensure_sentiment_analyzer()
    analyzed_payload = analyze_comments(raw_comments, analyzer)
    analysis_result = build_analysis_result(input_path, scrape_metadata, analyzed_payload)

    output_dir = input_path.parent
    analysis_json_path = output_dir / "analysis_results.json"
    report_html_path = output_dir / "index.html"

    analysis_json_path.write_text(
        json.dumps(analysis_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_html_path.write_text(
        render_html_report(analysis_result),
        encoding="utf-8",
    )

    print(f"Saved analysis JSON: {analysis_json_path}")
    print(f"Saved HTML report:   {report_html_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

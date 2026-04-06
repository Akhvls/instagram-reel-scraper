#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / ".model-cache"
REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))

try:
    from detoxify import Detoxify
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    Detoxify = None  # type: ignore[assignment]
    DETOXIFY_IMPORT_ERROR = exc
else:
    DETOXIFY_IMPORT_ERROR = None

TOXICITY_FIELDS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)
TOXICITY_SUBCATEGORIES = TOXICITY_FIELDS[1:]
TOXIC_THRESHOLD = 0.5
MILD_THRESHOLD = 0.3
TRIGGER_THRESHOLD = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Instagram reel comments JSON with Detoxify and generate "
            "analysis_results.json plus index.html inside a timestamped report folder."
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


def ensure_detector() -> Any:
    if Detoxify is None:
        raise RuntimeError(
            "detoxify is not installed. Install dependencies first with "
            "'pip install -r requirements.txt'."
        ) from DETOXIFY_IMPORT_ERROR

    try:
        return Detoxify("unbiased")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Failed to load Detoxify('unbiased'). Ensure the package is installed "
            "and the model can be downloaded in this environment."
        ) from exc


def coerce_number(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def classify_toxicity(toxicity_score: float) -> str:
    if toxicity_score >= TOXIC_THRESHOLD:
        return "toxic"
    if toxicity_score >= MILD_THRESHOLD:
        return "mild"
    return "clean"


def infer_shortcode(scrape_metadata: dict[str, Any], input_path: Path) -> str:
    shortcode = scrape_metadata.get("shortcode")
    if shortcode:
        return str(shortcode)

    match = re.fullmatch(r"comments_([^_]+)(?:_\d{8}_\d{6})?", input_path.stem)
    if match:
        return match.group(1)

    return "unknown"


def build_report_dir(shortcode: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(REPORT_TIMESTAMP_FORMAT)
    report_dir = REPORTS_DIR / f"{shortcode}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def is_inside_reports(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPORTS_DIR.resolve())
        return True
    except ValueError:
        return False


def resolve_report_dir(input_path: Path, shortcode: str) -> Path:
    parent_dir = input_path.parent
    if is_inside_reports(parent_dir):
        parent_dir.mkdir(parents=True, exist_ok=True)
        return parent_dir
    return build_report_dir(shortcode)


def triggered_subcategories(comment: dict[str, Any]) -> list[str]:
    triggered: list[str] = []
    for field_name in TOXICITY_SUBCATEGORIES:
        value = coerce_number(comment.get(field_name))
        if value is not None and value >= TRIGGER_THRESHOLD:
            triggered.append(field_name)
    return triggered


def comment_snapshot(comment: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
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
        "toxicity_label": comment.get("toxicity_label"),
        "triggered_subcategories": comment.get("triggered_subcategories") or [],
    }
    for field_name in TOXICITY_FIELDS:
        snapshot[field_name] = comment.get(field_name)
    return snapshot


def top_toxic_comments(
    comments: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    def score(comment: dict[str, Any], field_name: str) -> float:
        value = coerce_number(comment.get(field_name))
        return 0.0 if value is None else value

    ranked = sorted(
        comments,
        key=lambda comment: (
            score(comment, "toxicity"),
            score(comment, "insult"),
            score(comment, "severe_toxicity"),
            score(comment, "obscene"),
            score(comment, "identity_attack"),
            score(comment, "threat"),
            score(comment, "sexual_explicit"),
        ),
        reverse=True,
    )
    return [comment_snapshot(comment) for comment in ranked[:limit]]


def analyze_comments(
    raw_comments: list[dict[str, Any]],
    detector: Any,
) -> dict[str, Any]:
    analyzed_comments: list[dict[str, Any]] = []
    distribution_counts = {
        "toxic": 0,
        "mild": 0,
        "clean": 0,
    }

    total_comments = len(raw_comments)
    if total_comments:
        print(f"Scoring toxicity for {total_comments} comments...", flush=True)

    for index, raw_comment in enumerate(raw_comments, start=1):
        comment = dict(raw_comment)
        text = comment.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        try:
            scores = detector.predict(text)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            comment_id = comment.get("comment_id") or "unknown"
            raise RuntimeError(f"Detoxify prediction failed for comment {comment_id}.") from exc

        for field_name in TOXICITY_FIELDS:
            score = coerce_number(scores.get(field_name))
            comment[field_name] = round_float(score if score is not None else 0.0)

        toxicity_score = float(comment["toxicity"])
        comment["toxicity_label"] = classify_toxicity(toxicity_score)
        comment["triggered_subcategories"] = triggered_subcategories(comment)

        analyzed_comments.append(comment)
        distribution_counts[comment["toxicity_label"]] += 1

        if index == 1 or index % 250 == 0 or index == total_comments:
            print(f"Processed {index}/{total_comments} comments", flush=True)

    total_comments = len(analyzed_comments)
    distribution = {}
    for label in ("toxic", "mild", "clean"):
        count = distribution_counts[label]
        percentage = 0.0 if total_comments == 0 else round((count / total_comments) * 100, 2)
        distribution[label] = {
            "count": count,
            "percentage": percentage,
        }

    return {
        "comments": analyzed_comments,
        "summary": {
            "total_comments": total_comments,
            "model": "Detoxify('unbiased')",
            "toxicity_label_thresholds": {
                "toxic": TOXIC_THRESHOLD,
                "mild": MILD_THRESHOLD,
                "clean_below": MILD_THRESHOLD,
            },
            "subcategory_trigger_threshold": TRIGGER_THRESHOLD,
            "toxicity_distribution": distribution,
        },
        "top_comments": {
            "most_toxic": top_toxic_comments(analyzed_comments, limit=10),
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


def render_metadata_value(label: str, value: Any) -> str:
    if label == "Reel URL" and value:
        safe_value = html.escape(str(value), quote=True)
        return f'<a href="{safe_value}">{html.escape(str(value))}</a>'
    return html.escape(format_number(value))


def render_metadata_section(metadata_rows: list[tuple[str, Any]]) -> str:
    return "\n".join(
        [
            '<section class="panel">',
            "  <h2>Scrape Metadata</h2>",
            '  <dl class="metadata-grid">',
            *[
                f"    <dt>{html.escape(label)}</dt><dd>{render_metadata_value(label, value)}</dd>"
                for label, value in metadata_rows
            ],
            "  </dl>",
            "</section>",
        ]
    )


def render_toxic_comment_list(comments: list[dict[str, Any]]) -> str:
    items: list[str] = []

    for comment in comments:
        username = html.escape(comment.get("username") or "unknown")
        text = html.escape(str(comment.get("text") or "").strip() or "[no text]")
        permalink = comment.get("permalink")
        toxicity = coerce_number(comment.get("toxicity"))
        created_at = comment.get("created_at_iso")
        label = html.escape(str(comment.get("toxicity_label") or "unknown"))

        link_html = ""
        if permalink:
            safe_permalink = html.escape(str(permalink), quote=True)
            link_html = f' <a href="{safe_permalink}">permalink</a>'

        triggered = comment.get("triggered_subcategories") or []
        triggered_text = ", ".join(
            f"{field.replace('_', ' ')} ({float(comment[field]):.4f})"
            for field in triggered
            if coerce_number(comment.get(field)) is not None
        )
        if not triggered_text:
            triggered_text = f"none >= {TRIGGER_THRESHOLD:.2f}"

        footer_parts = [
            f"toxicity {toxicity:.4f}" if toxicity is not None else "toxicity n/a",
            label,
            "reply" if comment.get("is_reply") else "top-level",
        ]
        if created_at:
            footer_parts.append(html.escape(str(created_at)))

        items.append(
            "\n".join(
                [
                    '<li class="comment-item">',
                    f'  <div class="comment-meta">@{username}{link_html}</div>',
                    f'  <blockquote>{text}</blockquote>',
                    f'  <div class="comment-subcats"><strong>Triggered subcategories:</strong> {html.escape(triggered_text)}</div>',
                    f'  <div class="comment-footer">{" | ".join(footer_parts)}</div>',
                    "</li>",
                ]
            )
        )

    if not items:
        items.append('<li class="comment-item empty">No comments available.</li>')

    return "\n".join(
        [
            '<section class="panel">',
            "  <h2>Most Toxic Comments</h2>",
            '  <ol class="comment-list">',
            *items,
            "  </ol>",
            "</section>",
        ]
    )


def render_warnings_section(warnings: list[str]) -> str:
    if not warnings:
        return ""

    warning_items = [f"<li>{html.escape(str(warning))}</li>" for warning in warnings]
    return "\n".join(
        [
            '<section class="panel">',
            "  <h2>Warnings</h2>",
            '  <ul class="warning-list">',
            *warning_items,
            "  </ul>",
            "</section>",
        ]
    )


def render_html_report(analysis: dict[str, Any]) -> str:
    scrape_metadata = analysis.get("scrape_metadata") or {}
    summary = analysis["summary"]
    top_comments = analysis["top_comments"]
    distribution = summary["toxicity_distribution"]

    toxic_pct = distribution["toxic"]["percentage"]
    mild_pct = distribution["mild"]["percentage"]
    clean_pct = distribution["clean"]["percentage"]

    metadata_rows = [
        ("Reel URL", analysis.get("reel_url")),
        ("Shortcode", analysis.get("shortcode")),
        ("Media ID", analysis.get("media_id")),
        ("Source JSON", analysis.get("source_file")),
        ("Report Directory", analysis.get("report_directory")),
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
        ("Model", summary.get("model")),
    ]

    metadata_html = render_metadata_section(metadata_rows)
    warnings_html = render_warnings_section(scrape_metadata.get("warnings") or [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Instagram Reel Comment Toxicity Report</title>
  <style>
    :root {{
      --paper: #f5f1e8;
      --ink: #201a16;
      --muted: #665e57;
      --line: #cfc6b7;
      --toxic: #8a4d47;
      --mild: #c1a44b;
      --clean: #4f7f51;
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

    .segment-toxic {{
      background: var(--toxic);
      width: {toxic_pct:.2f}%;
    }}

    .segment-mild {{
      background: var(--mild);
      width: {mild_pct:.2f}%;
    }}

    .segment-clean {{
      background: var(--clean);
      width: {clean_pct:.2f}%;
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

    .comment-subcats {{
      color: var(--muted);
      font-size: 0.94rem;
      margin-bottom: 6px;
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
</head>
<body>
  <main>
    <header>
      <h1>Instagram Reel Comment Toxicity Report</h1>
      <p class="lede">A minimal static summary of Detoxify unbiased toxicity scores across the scraped reel comment set.</p>
    </header>

{metadata_html}

    <section class="panel">
      <h2>Toxicity Distribution</h2>
      <div class="distribution-bar" aria-label="Toxicity distribution bar">
        <div class="segment-toxic"></div>
        <div class="segment-mild"></div>
        <div class="segment-clean"></div>
      </div>
      <div class="distribution-legend">
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--toxic);"></span>Toxic: {format_percentage(toxic_pct)} ({distribution["toxic"]["count"]})</span>
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--mild);"></span>Mild: {format_percentage(mild_pct)} ({distribution["mild"]["count"]})</span>
        <span class="legend-chip"><span class="legend-swatch" style="background: var(--clean);"></span>Clean: {format_percentage(clean_pct)} ({distribution["clean"]["count"]})</span>
      </div>
      <p class="note">Labels use the overall toxicity score: toxic ≥ {TOXIC_THRESHOLD:.2f}, mild ≥ {MILD_THRESHOLD:.2f}, clean below {MILD_THRESHOLD:.2f}. Subcategories are shown as triggered when their score is at least {TRIGGER_THRESHOLD:.2f}.</p>
    </section>

{render_toxic_comment_list(top_comments["most_toxic"])}

{warnings_html}
  </main>
</body>
</html>
"""


def build_analysis_result(
    input_path: Path,
    scrape_metadata: dict[str, Any],
    analyzed_payload: dict[str, Any],
    report_dir: Path,
    shortcode: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path.name),
        "report_directory": str(report_dir),
        "reel_url": scrape_metadata.get("reel_url"),
        "shortcode": shortcode,
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
    shortcode = infer_shortcode(scrape_metadata, input_path)
    detector = ensure_detector()
    analyzed_payload = analyze_comments(raw_comments, detector)
    report_dir = resolve_report_dir(input_path, shortcode)
    analysis_result = build_analysis_result(
        input_path=input_path,
        scrape_metadata=scrape_metadata,
        analyzed_payload=analyzed_payload,
        report_dir=report_dir,
        shortcode=shortcode,
    )

    analysis_json_path = report_dir / "analysis_results.json"
    report_html_path = report_dir / "index.html"

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

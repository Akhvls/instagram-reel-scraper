#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

BASE_DIR = Path(__file__).resolve().parent
LOCAL_BROWSER_DIR = BASE_DIR / ".playwright-browsers"
PROFILE_DIR = BASE_DIR / "browser-profile"

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(LOCAL_BROWSER_DIR))

from playwright.sync_api import Error, Page, Response, TimeoutError as PlaywrightTimeoutError, sync_playwright

COMMENT_ICON_SELECTOR = "img[alt='Comment'], svg[aria-label='Comment']"
DEFAULT_COMMENT_QUERY_NAME = "PolarisPostCommentsContainerQuery"
HTTP_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_REQUEST_ATTEMPTS = 5
DEFAULT_INITIAL_BACKOFF_MS = 1_000
DEFAULT_MAX_BACKOFF_MS = 8_000
REPLY_REQUEST_ATTEMPTS = 2
REPLY_INITIAL_BACKOFF_MS = 750
REPLY_MAX_BACKOFF_MS = 750
RETRYABLE_ERROR_SNIPPETS = (
    "please wait a few minutes",
    "try again later",
    "temporarily blocked",
    "challenge_required",
    "feedback_required",
    "rate limit",
)

DISMISS_BUTTON_PATTERNS = [
    re.compile(r"^not now$", re.I),
    re.compile(r"^only allow essential cookies$", re.I),
    re.compile(r"^allow all cookies$", re.I),
    re.compile(r"^close$", re.I),
]

FETCH_JSON_JS = """
async ({ path, params }) => {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value === null || value === undefined) {
      continue;
    }
    qs.set(key, typeof value === "string" ? value : JSON.stringify(value));
  }

  const url = qs.toString() ? `${path}?${qs.toString()}` : path;
  const response = await fetch(url, {
    method: "GET",
    credentials: "include",
  });

  return {
    ok: response.ok,
    status: response.status,
    url,
    text: await response.text(),
  };
}
"""

READ_COMMENT_COUNT_JS = """
() => {
  const buttons = Array.from(document.querySelectorAll("button"));
  for (const button of buttons) {
    if (!button.querySelector("img[alt='Comment'], svg[aria-label='Comment']")) {
      continue;
    }

    const raw = (button.innerText || button.textContent || "").replace(/\\s+/g, " ").trim();
    if (!raw) {
      continue;
    }

    const match = raw.match(/([\\d.,]+\\s*[kKmM]?)/);
    if (match) {
      return match[1];
    }
  }

  return null;
}
"""


def normalize_reel_url(raw_url: str) -> str | None:
    reel_url = raw_url.strip().strip("\"'")

    if reel_url.startswith("/reel/") or reel_url.startswith("/reels/"):
        reel_url = f"https://www.instagram.com{reel_url}"
    elif reel_url.startswith("www.instagram.com/"):
        reel_url = f"https://{reel_url}"
    elif reel_url.startswith("instagram.com/"):
        reel_url = f"https://www.{reel_url}"

    if not reel_url.startswith(("http://", "https://")):
        return None

    parsed = urlparse(reel_url)
    host = parsed.netloc.lower()
    if not host.endswith("instagram.com"):
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2 or path_parts[0] not in {"reel", "reels"}:
        return None

    normalized_path = f"/reel/{path_parts[1]}/"
    return urlunparse(
        (
            "https",
            "www.instagram.com",
            normalized_path,
            "",
            parsed.query,
            parsed.fragment,
        )
    )


def prompt_for_reel_url() -> str:
    while True:
        raw_url = input("Paste the Instagram reel URL: ").strip()
        if not raw_url:
            print("A reel URL is required.")
            continue

        reel_url = normalize_reel_url(raw_url)
        if reel_url is None:
            print(
                "Paste a reel URL like https://www.instagram.com/reel/... "
                "or https://www.instagram.com/reels/..."
            )
            continue

        return reel_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Instagram reel comments by reusing your persistent browser "
            "session and Instagram's own comment endpoints."
        )
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Instagram reel URL. If omitted, the script prompts for one.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional JSON output path. A CSV file is written alongside it.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headlessly. Headed mode is more reliable for fresh logins.",
    )
    parser.add_argument(
        "--wait-for-login-seconds",
        type=int,
        default=180,
        help="How long to wait for login/challenge completion if Instagram blocks comments.",
    )
    parser.add_argument(
        "--max-comment-pages",
        type=int,
        default=400,
        help="Safety cap for top-level comment pages.",
    )
    parser.add_argument(
        "--max-reply-pages",
        type=int,
        default=50,
        help="Safety cap for reply pages on a single parent comment.",
    )
    parser.add_argument(
        "--max-no-growth-pages",
        type=int,
        default=3,
        help="Stop after this many cursor pages add no new rows.",
    )
    parser.add_argument(
        "--checkpoint-every-comment-pages",
        type=int,
        default=10,
        help="Write a partial checkpoint after this many top-level pages.",
    )
    parser.add_argument(
        "--checkpoint-every-reply-parents",
        type=int,
        default=25,
        help="Write a partial checkpoint after this many reply threads.",
    )
    parser.add_argument(
        "--throttle-ms",
        type=int,
        default=250,
        help="Delay between API pages to reduce rate-limit risk.",
    )
    return parser.parse_args()


def shortcode_from_url(reel_url: str) -> str:
    path_parts = [part for part in urlparse(reel_url).path.split("/") if part]
    return path_parts[1]


def output_paths(reel_url: str, requested_output: Path | None) -> tuple[Path, Path]:
    if requested_output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_output = BASE_DIR / f"comments_{shortcode_from_url(reel_url)}_{timestamp}.json"
    elif requested_output.suffix.lower() != ".json":
        requested_output = requested_output.with_suffix(".json")

    return requested_output, requested_output.with_suffix(".csv")


def partial_output_paths(final_json_path: Path) -> tuple[Path, Path]:
    partial_json_path = final_json_path.with_name(f"{final_json_path.stem}.partial.json")
    return partial_json_path, partial_json_path.with_suffix(".csv")


def parse_instagram_count(raw_value: str | None) -> int | None:
    if not raw_value:
        return None

    text = raw_value.strip().replace(",", "").replace(" ", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", text)
    if not match:
        return None

    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000

    return int(round(number))


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def epoch_to_iso(epoch_value: Any) -> str | None:
    if epoch_value in (None, ""):
        return None

    try:
        return datetime.fromtimestamp(int(epoch_value), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def parse_request_fields(response: Response) -> dict[str, str]:
    parsed = parse_qs(response.request.post_data or "")
    return {key: values[-1] for key, values in parsed.items()}


def graphql_query_name(response: Response) -> str | None:
    try:
        return parse_request_fields(response).get("fb_api_req_friendly_name")
    except Error:
        return None


def dismiss_dialogs(page: Page) -> None:
    buttons = page.locator("button, div[role='button']")

    for pattern in DISMISS_BUTTON_PATTERNS:
        while True:
            target = buttons.filter(has_text=pattern).first
            try:
                if not target.is_visible(timeout=400):
                    break
                target.click(timeout=1_500)
                page.wait_for_timeout(400)
            except Error:
                break


def click_comment_button(page: Page) -> bool:
    strategies = [
        page.get_by_role("button", name=re.compile(r"^Comment\s+", re.I)).first,
        page.locator("button").filter(has=page.locator(COMMENT_ICON_SELECTOR)).first,
    ]

    for target in strategies:
        try:
            target.wait_for(state="visible", timeout=5_000)
            target.click(timeout=5_000)
            return True
        except Error:
            continue

    try:
        clicked = page.evaluate(
            """
            () => {
              const buttons = Array.from(document.querySelectorAll("button"));
              const visible = buttons.filter((button) => {
                const rect = button.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              });

              for (const button of visible) {
                if (!button.querySelector("img[alt='Comment'], svg[aria-label='Comment']")) {
                  continue;
                }
                button.click();
                return true;
              }

              return false;
            }
            """
        )
        return bool(clicked)
    except Error:
        return False


def wait_for_comment_entry(page: Page, timeout_ms: int) -> bool:
    deadline = datetime.now().timestamp() + (timeout_ms / 1000)

    while datetime.now().timestamp() < deadline:
        dismiss_dialogs(page)
        if click_comment_button(page):
            return True
        page.wait_for_timeout(1_500)

    return False


def capture_query_fields(page: Page, query_name: str, trigger, timeout_ms: int = 20_000) -> dict[str, str]:
    def predicate(response: Response) -> bool:
        return "/graphql/query" in response.url and graphql_query_name(response) == query_name

    with page.expect_response(predicate, timeout=timeout_ms) as response_info:
        trigger()

    return parse_request_fields(response_info.value)


def cursor_key(cursor: Any) -> str | None:
    if cursor is None:
        return None

    if isinstance(cursor, (dict, list)):
        return json.dumps(cursor, sort_keys=True, separators=(",", ":"))

    return str(cursor)


def request_cursor_marker(cursor: Any) -> str:
    key = cursor_key(cursor)
    if key in (None, ""):
        return "__root__"
    return key


def is_retryable_message(message: str | None) -> bool:
    if not message:
        return False

    lowered = message.lower()
    return any(snippet in lowered for snippet in RETRYABLE_ERROR_SNIPPETS)


def instagram_error_message(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    if payload.get("status") == "fail":
        return payload.get("message") or payload.get("error_type") or payload.get("error")

    if payload.get("error_type") or payload.get("error"):
        return payload.get("message") or payload.get("error_type") or payload.get("error")

    return None


def fetch_json(
    page: Page,
    path: str,
    params: dict[str, Any],
    description: str,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
    initial_backoff_ms: int = DEFAULT_INITIAL_BACKOFF_MS,
    max_backoff_ms: int = DEFAULT_MAX_BACKOFF_MS,
) -> dict[str, Any]:
    backoff_ms = initial_backoff_ms
    last_error = f"{description} failed."

    for attempt in range(1, attempts + 1):
        retryable = False
        try:
            result = page.evaluate(
                FETCH_JSON_JS,
                {
                    "path": path,
                    "params": params,
                },
            )
        except Error as exc:
            last_error = f"{description} browser fetch failed: {exc}"
            retryable = True
        else:
            status = int(result.get("status") or 0)
            payload_text = result.get("text") or ""
            if payload_text.startswith("for (;;);"):
                payload_text = payload_text[len("for (;;);") :]

            try:
                payload = json.loads(payload_text) if payload_text else {}
            except json.JSONDecodeError:
                sample = payload_text[:200].replace("\n", " ")
                last_error = (
                    f"{description} returned non-JSON data with HTTP {status}. "
                    f"Sample: {sample!r}"
                )
                retryable = status in HTTP_RETRY_STATUS_CODES or is_retryable_message(sample)
            else:
                message = instagram_error_message(payload)
                if 200 <= status < 300 and message is None:
                    return payload if isinstance(payload, dict) else {"data": payload}

                sample = message or payload_text[:200].replace("\n", " ")
                last_error = f"{description} failed with HTTP {status}: {sample}"
                retryable = status in HTTP_RETRY_STATUS_CODES or is_retryable_message(sample)

        if not retryable or attempt >= attempts:
            break

        print(
            f"[retry] {description} attempt {attempt}/{attempts} failed. "
            f"Waiting {backoff_ms / 1000:.1f}s before retry.",
            file=sys.stderr,
            flush=True,
        )
        page.wait_for_timeout(backoff_ms)
        backoff_ms = min(backoff_ms * 2, max_backoff_ms)

    raise RuntimeError(last_error)


def fetch_comments_page(page: Page, media_id: str, cursor: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "can_support_threading": "true",
        "permalink_enabled": "true",
    }
    if cursor is not None:
        params["min_id"] = cursor

    return fetch_json(
        page,
        f"/api/v1/media/{media_id}/comments/",
        params,
        "top-level comments request",
    )


def fetch_child_comments_page(page: Page, media_id: str, parent_comment_id: str, cursor: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if cursor is not None:
        params["max_id"] = cursor

    return fetch_json(
        page,
        f"/api/v1/media/{media_id}/comments/{parent_comment_id}/child_comments/",
        params,
        f"reply request for parent {parent_comment_id}",
        attempts=REPLY_REQUEST_ATTEMPTS,
        initial_backoff_ms=REPLY_INITIAL_BACKOFF_MS,
        max_backoff_ms=REPLY_MAX_BACKOFF_MS,
    )


def dedupe_nodes(existing: dict[str, dict[str, Any]], nodes: list[dict[str, Any]]) -> int:
    added = 0

    for node in nodes:
        comment_id = str(node.get("pk") or "")
        if not comment_id or comment_id in existing:
            continue
        existing[comment_id] = node
        added += 1

    return added


def read_reported_comment_count_from_dom(page: Page) -> int | None:
    try:
        raw = page.evaluate(READ_COMMENT_COUNT_JS)
    except Error:
        return None

    return parse_instagram_count(raw)


def fetch_top_level_comments(
    page: Page,
    media_id: str,
    max_comment_pages: int,
    max_no_growth_pages: int,
    throttle_ms: int,
    checkpoint_every_pages: int,
    checkpoint_callback,
) -> tuple[list[dict[str, Any]], int | None, list[str]]:
    ordered_nodes: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    seen_request_cursors: set[str] = set()
    cursor: Any = None
    reported_comment_count: int | None = None
    no_growth_pages = 0

    for page_number in range(1, max_comment_pages + 1):
        request_cursor_key = request_cursor_marker(cursor)
        if request_cursor_key in seen_request_cursors:
            warning = (
                f"Top-level cursor loop detected after page {page_number - 1}. "
                "Stopping pagination to avoid an endless loop."
            )
            warnings.append(warning)
            print(f"[top-level] {warning}", file=sys.stderr, flush=True)
            break
        seen_request_cursors.add(request_cursor_key)

        payload = fetch_comments_page(page, media_id, cursor)
        if reported_comment_count is None:
            reported_comment_count = safe_int(payload.get("comment_count"))

        batch = payload.get("comments") or []
        added = dedupe_nodes(ordered_nodes, batch)
        next_cursor = payload.get("next_min_id")

        if added == 0:
            no_growth_pages += 1
        else:
            no_growth_pages = 0

        print(
            f"[top-level] page {page_number}: batch={len(batch)} added={added} "
            f"total={len(ordered_nodes)} next_cursor={'yes' if next_cursor else 'no'}",
            flush=True,
        )

        if checkpoint_every_pages > 0 and (
            page_number == 1 or page_number % checkpoint_every_pages == 0 or not next_cursor
        ):
            checkpoint_callback(
                list(ordered_nodes.values()),
                {},
                "top_level",
                f"Fetched {page_number} top-level pages.",
                warnings,
            )

        if not next_cursor:
            break

        if no_growth_pages >= max_no_growth_pages:
            warning = (
                f"Top-level pagination added no new comments for {max_no_growth_pages} pages. "
                "Stopping to avoid looping forever."
            )
            warnings.append(warning)
            print(f"[top-level] {warning}", file=sys.stderr, flush=True)
            break

        if throttle_ms > 0:
            page.wait_for_timeout(throttle_ms)
        cursor = next_cursor
    else:
        warning = (
            f"Reached the --max-comment-pages cap ({max_comment_pages}) before Instagram "
            "stopped returning cursors."
        )
        warnings.append(warning)
        print(f"[top-level] {warning}", file=sys.stderr, flush=True)

    if reported_comment_count is None:
        reported_comment_count = read_reported_comment_count_from_dom(page)

    return list(ordered_nodes.values()), reported_comment_count, warnings


def fetch_replies_for_comment(
    page: Page,
    media_id: str,
    parent_comment: dict[str, Any],
    max_reply_pages: int,
    max_no_growth_pages: int,
    throttle_ms: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    parent_comment_id = str(parent_comment.get("pk") or "")
    child_comment_count = safe_int(parent_comment.get("child_comment_count")) or 0
    if not parent_comment_id or child_comment_count <= 0:
        return [], warnings

    replies: dict[str, dict[str, Any]] = {}
    preview_child_comments = parent_comment.get("preview_child_comments") or []
    dedupe_nodes(replies, preview_child_comments)
    if len(replies) >= child_comment_count:
        return list(replies.values()), warnings

    seen_request_cursors: set[str] = set()
    cursor: Any = ""
    no_growth_pages = 0

    for page_number in range(1, max_reply_pages + 1):
        request_cursor_key = request_cursor_marker(cursor)
        if request_cursor_key in seen_request_cursors and page_number > 1:
            warning = (
                f"Reply cursor loop detected for parent {parent_comment_id}. "
                "Stopping that reply thread."
            )
            warnings.append(warning)
            break
        seen_request_cursors.add(request_cursor_key)

        payload = fetch_child_comments_page(page, media_id, parent_comment_id, cursor)
        batch = payload.get("child_comments") or []
        added = dedupe_nodes(replies, batch)
        next_cursor = payload.get("next_max_child_cursor")

        if added == 0:
            no_growth_pages += 1
        else:
            no_growth_pages = 0

        if page_number > 1 or next_cursor or len(replies) < child_comment_count:
            print(
                f"[reply {parent_comment_id}] page {page_number}: batch={len(batch)} "
                f"added={added} total={len(replies)} next_cursor={'yes' if next_cursor else 'no'}",
                flush=True,
            )

        if not next_cursor:
            break

        if no_growth_pages >= max_no_growth_pages:
            warning = (
                f"Reply pagination for parent {parent_comment_id} added no new comments for "
                f"{max_no_growth_pages} pages. Stopping that thread."
            )
            warnings.append(warning)
            break

        if throttle_ms > 0:
            page.wait_for_timeout(throttle_ms)
        cursor = next_cursor
    else:
        warnings.append(
            f"Reached the --max-reply-pages cap ({max_reply_pages}) for parent {parent_comment_id}."
        )

    return list(replies.values()), warnings


def fetch_all_replies(
    page: Page,
    media_id: str,
    top_level_comments: list[dict[str, Any]],
    max_reply_pages: int,
    max_no_growth_pages: int,
    throttle_ms: int,
    checkpoint_every_reply_parents: int,
    checkpoint_callback,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    parents = [
        comment
        for comment in top_level_comments
        if (safe_int(comment.get("child_comment_count")) or 0) > 0 and comment.get("pk")
    ]
    if not parents:
        return {}, []

    print(
        f"[replies] fetching replies for {len(parents)} top-level comments that have replies.",
        flush=True,
    )

    replies_by_parent: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    total_replies = 0

    for index, parent_comment in enumerate(parents, start=1):
        parent_comment_id = str(parent_comment.get("pk"))

        try:
            replies, reply_warnings = fetch_replies_for_comment(
                page=page,
                media_id=media_id,
                parent_comment=parent_comment,
                max_reply_pages=max_reply_pages,
                max_no_growth_pages=max_no_growth_pages,
                throttle_ms=throttle_ms,
            )
        except RuntimeError as exc:
            reply_warnings = [f"Replies for parent {parent_comment_id} failed: {exc}"]
            replies = []

        if replies:
            replies_by_parent[parent_comment_id] = replies
            total_replies += len(replies)

        warnings.extend(reply_warnings)

        if index == 1 or index % 25 == 0 or index == len(parents):
            print(
                f"[replies] processed {index}/{len(parents)} parents, total replies={total_replies}",
                flush=True,
            )

        if checkpoint_every_reply_parents > 0 and (
            index == 1
            or index % checkpoint_every_reply_parents == 0
            or index == len(parents)
        ):
            checkpoint_callback(
                top_level_comments,
                replies_by_parent,
                "replies",
                f"Processed {index} of {len(parents)} reply threads.",
                warnings,
            )

    return replies_by_parent, warnings


def normalize_comment_node(
    node: dict[str, Any],
    shortcode: str,
    order: int,
    top_level_order: int,
    is_reply: bool,
    reply_order: int | None = None,
    parent_comment_id: str | None = None,
) -> dict[str, Any]:
    user = node.get("user") or {}
    comment_id = str(node.get("pk")) if node.get("pk") is not None else None
    parent_id = parent_comment_id or node.get("parent_comment_id")

    return {
        "order": order,
        "top_level_order": top_level_order,
        "reply_order": reply_order,
        "comment_id": comment_id,
        "parent_comment_id": str(parent_id) if parent_id not in (None, "") else None,
        "username": user.get("username"),
        "user_id": str(user.get("pk") or user.get("id")) if (user.get("pk") or user.get("id")) else None,
        "is_verified": user.get("is_verified"),
        "text": node.get("text"),
        "created_at_epoch": node.get("created_at"),
        "created_at_iso": epoch_to_iso(node.get("created_at")),
        "like_count": node.get("comment_like_count"),
        "reply_count": node.get("child_comment_count"),
        "is_reply": is_reply,
        "permalink": (
            f"https://www.instagram.com/p/{shortcode}/c/{comment_id}/"
            if comment_id
            else None
        ),
    }


def flatten_comments(
    shortcode: str,
    top_level_comments: list[dict[str, Any]],
    replies_by_parent: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    flat_comments: list[dict[str, Any]] = []
    order = 1

    for top_level_order, comment in enumerate(top_level_comments, start=1):
        normalized_comment = normalize_comment_node(
            node=comment,
            shortcode=shortcode,
            order=order,
            top_level_order=top_level_order,
            is_reply=False,
        )
        flat_comments.append(normalized_comment)
        order += 1

        parent_comment_id = normalized_comment["comment_id"]
        replies = replies_by_parent.get(parent_comment_id or "", [])
        for reply_order, reply in enumerate(replies, start=1):
            flat_comments.append(
                normalize_comment_node(
                    node=reply,
                    shortcode=shortcode,
                    order=order,
                    top_level_order=top_level_order,
                    is_reply=True,
                    reply_order=reply_order,
                    parent_comment_id=parent_comment_id,
                )
            )
            order += 1

    return flat_comments


def save_results(
    reel_url: str,
    shortcode: str,
    media_id: str,
    reported_comment_count: int | None,
    comments: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    warnings: list[str] | None = None,
    is_partial: bool = False,
    stage: str | None = None,
    note: str | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    top_level_comments = [comment for comment in comments if not comment["is_reply"]]
    reply_comments = [comment for comment in comments if comment["is_reply"]]

    payload = {
        "reel_url": reel_url,
        "shortcode": shortcode,
        "media_id": media_id,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "reported_top_level_comment_count": reported_comment_count,
        "exported_top_level_comment_count": len(top_level_comments),
        "exported_reply_count": len(reply_comments),
        "exported_total_count": len(comments),
        "is_partial": is_partial,
        "stage": stage,
        "note": note,
        "warnings": warnings or [],
        "comments": comments,
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "order",
        "top_level_order",
        "reply_order",
        "comment_id",
        "parent_comment_id",
        "username",
        "user_id",
        "is_verified",
        "text",
        "created_at_epoch",
        "created_at_iso",
        "like_count",
        "reply_count",
        "is_reply",
        "permalink",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comment in comments:
            writer.writerow({field: comment.get(field) for field in fieldnames})


def open_reel(page: Page, reel_url: str) -> None:
    page.goto(reel_url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(4_000)
    dismiss_dialogs(page)


def get_media_id_from_comment_entry(page: Page, reel_url: str, wait_for_login_seconds: int) -> str:
    def trigger_comment_capture() -> None:
        if not click_comment_button(page):
            raise RuntimeError("Comment button click failed.")

    open_reel(page, reel_url)

    try:
        fields = capture_query_fields(
            page,
            DEFAULT_COMMENT_QUERY_NAME,
            trigger_comment_capture,
            timeout_ms=15_000,
        )
    except (PlaywrightTimeoutError, RuntimeError):
        print(
            "Instagram did not expose comments immediately. "
            "If a login or challenge page opened in the browser, complete it now.",
            file=sys.stderr,
            flush=True,
        )
        if not wait_for_comment_entry(page, wait_for_login_seconds * 1000):
            raise RuntimeError(
                "Comments never became available. Make sure the Instagram session in "
                f"{PROFILE_DIR} is still logged in."
            )
        open_reel(page, reel_url)
        fields = capture_query_fields(
            page,
            DEFAULT_COMMENT_QUERY_NAME,
            trigger_comment_capture,
            timeout_ms=20_000,
        )

    try:
        variables = json.loads(fields["variables"])
        return str(variables["media_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Instagram changed the comment request format, so the media ID could not be read.") from exc


def scrape_reel_comments(
    page: Page,
    reel_url: str,
    json_path: Path,
    csv_path: Path,
    wait_for_login_seconds: int,
    max_comment_pages: int,
    max_reply_pages: int,
    max_no_growth_pages: int,
    checkpoint_every_comment_pages: int,
    checkpoint_every_reply_parents: int,
    throttle_ms: int,
) -> tuple[str, int | None, list[dict[str, Any]], list[str]]:
    shortcode = shortcode_from_url(reel_url)
    partial_json_path, partial_csv_path = partial_output_paths(json_path)
    media_id = get_media_id_from_comment_entry(page, reel_url, wait_for_login_seconds)

    def checkpoint_callback(
        top_level_comments: list[dict[str, Any]],
        replies_by_parent: dict[str, list[dict[str, Any]]],
        stage: str,
        note: str,
        warnings: list[str],
    ) -> None:
        flat_comments = flatten_comments(shortcode, top_level_comments, replies_by_parent)
        save_results(
            reel_url=reel_url,
            shortcode=shortcode,
            media_id=media_id,
            reported_comment_count=None,
            comments=flat_comments,
            json_path=partial_json_path,
            csv_path=partial_csv_path,
            warnings=warnings,
            is_partial=True,
            stage=stage,
            note=note,
        )
        print(
            f"[checkpoint] wrote {len(flat_comments)} rows to {partial_json_path}",
            flush=True,
        )

    top_level_comments, reported_comment_count, top_level_warnings = fetch_top_level_comments(
        page=page,
        media_id=media_id,
        max_comment_pages=max_comment_pages,
        max_no_growth_pages=max_no_growth_pages,
        throttle_ms=throttle_ms,
        checkpoint_every_pages=checkpoint_every_comment_pages,
        checkpoint_callback=checkpoint_callback,
    )

    def reply_checkpoint_callback(
        top_level_for_checkpoint: list[dict[str, Any]],
        replies_by_parent: dict[str, list[dict[str, Any]]],
        stage: str,
        note: str,
        warnings: list[str],
    ) -> None:
        flat_comments = flatten_comments(shortcode, top_level_for_checkpoint, replies_by_parent)
        save_results(
            reel_url=reel_url,
            shortcode=shortcode,
            media_id=media_id,
            reported_comment_count=reported_comment_count,
            comments=flat_comments,
            json_path=partial_json_path,
            csv_path=partial_csv_path,
            warnings=warnings,
            is_partial=True,
            stage=stage,
            note=note,
        )
        print(
            f"[checkpoint] wrote {len(flat_comments)} rows to {partial_json_path}",
            flush=True,
        )

    replies_by_parent, reply_warnings = fetch_all_replies(
        page=page,
        media_id=media_id,
        top_level_comments=top_level_comments,
        max_reply_pages=max_reply_pages,
        max_no_growth_pages=max_no_growth_pages,
        throttle_ms=throttle_ms,
        checkpoint_every_reply_parents=checkpoint_every_reply_parents,
        checkpoint_callback=reply_checkpoint_callback,
    )

    flat_comments = flatten_comments(
        shortcode=shortcode,
        top_level_comments=top_level_comments,
        replies_by_parent=replies_by_parent,
    )
    warnings = top_level_warnings + reply_warnings
    return media_id, reported_comment_count, flat_comments, warnings


def main() -> int:
    args = parse_args()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    if not LOCAL_BROWSER_DIR.exists():
        print(
            "Local Playwright browsers are missing. Run:\n"
            "PLAYWRIGHT_BROWSERS_PATH=.playwright-browsers "
            ".venv/bin/python -m playwright install chromium",
            file=sys.stderr,
        )
        return 1

    reel_url = normalize_reel_url(args.url) if args.url else prompt_for_reel_url()
    if reel_url is None:
        print("That is not a valid Instagram reel URL.", file=sys.stderr)
        return 1

    json_path, csv_path = output_paths(reel_url, args.output)
    partial_json_path, partial_csv_path = partial_output_paths(json_path)

    print(f"Scraping reel: {reel_url}", flush=True)
    print(f"Final JSON: {json_path}", flush=True)
    print(f"Final CSV:  {csv_path}", flush=True)
    print(f"Checkpoint JSON: {partial_json_path}", flush=True)
    print(f"Checkpoint CSV:  {partial_csv_path}", flush=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=args.headless,
        )

        try:
            page = context.pages[0] if context.pages else context.new_page()
            try:
                media_id, reported_comment_count, comments, warnings = scrape_reel_comments(
                    page=page,
                    reel_url=reel_url,
                    json_path=json_path,
                    csv_path=csv_path,
                    wait_for_login_seconds=args.wait_for_login_seconds,
                    max_comment_pages=args.max_comment_pages,
                    max_reply_pages=args.max_reply_pages,
                    max_no_growth_pages=args.max_no_growth_pages,
                    checkpoint_every_comment_pages=args.checkpoint_every_comment_pages,
                    checkpoint_every_reply_parents=args.checkpoint_every_reply_parents,
                    throttle_ms=args.throttle_ms,
                )
            except Exception:
                if partial_json_path.exists():
                    print(f"Partial JSON kept at: {partial_json_path}", file=sys.stderr, flush=True)
                    print(f"Partial CSV kept at:  {partial_csv_path}", file=sys.stderr, flush=True)
                raise
        finally:
            context.close()

    save_results(
        reel_url=reel_url,
        shortcode=shortcode_from_url(reel_url),
        media_id=media_id,
        reported_comment_count=reported_comment_count,
        comments=comments,
        json_path=json_path,
        csv_path=csv_path,
        warnings=warnings,
    )

    partial_json_path.unlink(missing_ok=True)
    partial_csv_path.unlink(missing_ok=True)

    top_level_comments = [comment for comment in comments if not comment["is_reply"]]
    reply_comments = [comment for comment in comments if comment["is_reply"]]

    print(
        f"Saved {len(top_level_comments)} top-level comments and "
        f"{len(reply_comments)} replies ({len(comments)} total rows) to:"
    )
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    if reported_comment_count is not None:
        print(f"Instagram showed about {reported_comment_count} top-level comments on the reel.")
        if len(top_level_comments) < reported_comment_count:
            print(
                "Instagram's logged-in web session only exposed "
                f"{len(top_level_comments)} of those top-level comments on this run."
            )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Closing browser...")
    except PlaywrightTimeoutError:
        print("Instagram took too long to respond. Try again in headed mode.", file=sys.stderr)
        raise SystemExit(1)
    except Error as exc:
        print(f"Playwright error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

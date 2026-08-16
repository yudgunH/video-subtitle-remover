import bisect
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from backend.tools.app_paths import get_data_path


def _normalize_text(text):
    return re.sub(r"\s+", "", (text or "").strip()).casefold()


def _box_center(box):
    xmin, xmax, ymin, ymax = box
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)


def _box_iou(a, b):
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / max(1, a_area + b_area - intersection)


def _box_is_excluded(box, exclusion_areas):
    """Return True when an OCR box belongs to a user-marked subtitle zone."""
    xmin, xmax, ymin, ymax = box
    cx, cy = _box_center(box)
    box_area = max(1, (xmax - xmin) * (ymax - ymin))
    for eymin, eymax, exmin, exmax in exclusion_areas or []:
        if exmin <= cx <= exmax and eymin <= cy <= eymax:
            return True
        iw = max(0, min(xmax, exmax) - max(xmin, exmin))
        ih = max(0, min(ymax, eymax) - max(ymin, eymin))
        if (iw * ih) / box_area >= 0.5:
            return True
    return False


@dataclass
class TranslationTrack:
    source_text: str
    start_frame: int
    end_frame: int
    keyframes: list = field(default_factory=list)
    best_score: float = 0.0
    translated_text: str = ""

    def add(self, frame_no, box, text, score):
        self.end_frame = frame_no
        self.keyframes.append((frame_no, tuple(box)))
        if score >= self.best_score:
            self.source_text = text
            self.best_score = score

    @property
    def last_box(self):
        return self.keyframes[-1][1]

    def box_at(self, frame_no):
        if len(self.keyframes) == 1:
            return self.keyframes[0][1]
        frame_numbers = [item[0] for item in self.keyframes]
        pos = bisect.bisect_left(frame_numbers, frame_no)
        if pos <= 0:
            return self.keyframes[0][1]
        if pos >= len(self.keyframes):
            return self.keyframes[-1][1]
        left_frame, left_box = self.keyframes[pos - 1]
        right_frame, right_box = self.keyframes[pos]
        span = max(1, right_frame - left_frame)
        ratio = (frame_no - left_frame) / span
        return tuple(round(a + (b - a) * ratio) for a, b in zip(left_box, right_box))


def build_translation_tracks(records_by_frame, exclusion_areas=None, sample_step=1, total_frames=None):
    """Associate sampled OCR records into temporally stable text tracks."""
    tracks = []
    max_gap = max(2, sample_step * 2)

    for frame_no in sorted(records_by_frame):
        used_tracks = set()
        for record in records_by_frame[frame_no]:
            text = (record.get("text") or "").strip()
            box = tuple(record.get("box") or ())
            score = float(record.get("score") or 0.0)
            if not text or len(box) != 4 or _box_is_excluded(box, exclusion_areas):
                continue

            normalized = _normalize_text(text)
            best_index = None
            best_rank = -1.0
            for index, track in enumerate(tracks):
                if index in used_tracks or frame_no - track.end_frame > max_gap:
                    continue
                similarity = SequenceMatcher(
                    None, normalized, _normalize_text(track.source_text)
                ).ratio()
                iou = _box_iou(box, track.last_box)
                cx, cy = _box_center(box)
                tx, ty = _box_center(track.last_box)
                width = max(1, box[1] - box[0], track.last_box[1] - track.last_box[0])
                height = max(1, box[3] - box[2], track.last_box[3] - track.last_box[2])
                close = abs(cx - tx) <= max(40, width) and abs(cy - ty) <= max(24, height)
                if similarity < 0.72 or (iou < 0.08 and not close):
                    continue
                rank = similarity * 2 + iou
                if rank > best_rank:
                    best_rank = rank
                    best_index = index

            if best_index is None:
                track = TranslationTrack(text, frame_no, frame_no, best_score=score)
                track.keyframes.append((frame_no, box))
                tracks.append(track)
                best_index = len(tracks) - 1
            else:
                tracks[best_index].add(frame_no, box, text, score)
            used_tracks.add(best_index)

    for track in tracks:
        track.start_frame = max(1, track.start_frame - sample_step + 1)
        track.end_frame = track.end_frame + sample_step
        if total_frames is not None:
            track.end_frame = min(total_frames, track.end_frame)
    return tracks


class NineRouterTranslator:
    """OpenAI-compatible 9Router translation client with a persistent cache."""

    _local_start_lock = threading.Lock()

    def __init__(self, base_url, api_key, model, target_language, timeout=90):
        self.base_url = (
            base_url or "https://9router.hbfstudio.site/v1"
        ).rstrip("/,")
        self.api_key = (api_key or "").strip()
        self.model = (model or "auto").strip()
        self.target_language = (target_language or "Vietnamese").strip()
        self.timeout = timeout
        self.session = requests.Session()
        self._cache_path = str(
            get_data_path("cache", "translation_cache.json", create_parent=True)
        )
        self._cache_lock = threading.Lock()
        self._cache = self._load_cache()

    @property
    def api_v1_url(self):
        if self.base_url.endswith("/chat/completions"):
            return self.base_url[:-len("/chat/completions")]
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    @property
    def chat_completions_url(self):
        return f"{self.api_v1_url}/chat/completions"

    @staticmethod
    def choose_translation_model(models):
        """Prefer a fast general model over reasoning/coding models for translation."""
        models = [model for model in models or [] if model]
        if not models:
            return None
        unsuitable_terms = ("thinking", "reasoning", "agentic", "code", "opus")
        ordinary = [
            model for model in models
            if not any(term in model.casefold() for term in unsuitable_terms)
        ] or models
        for preferred_term in (
            "haiku", "mini", "flash", "fast", "glm", "qwen", "deepseek"
        ):
            match = next(
                (model for model in ordinary if preferred_term in model.casefold()),
                None,
            )
            if match:
                return match
        return ordinary[0]

    def _cache_key(self, text):
        return "|".join((self.base_url, self.model, self.target_language, text))

    def _load_cache(self):
        try:
            with open(self._cache_path, "r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        temporary_path = f"{self._cache_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as cache_file:
            json.dump(self._cache, cache_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self._cache_path)

    @staticmethod
    def _extract_json(content):
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content = str(content or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("9Router returned no JSON object")
        return json.loads(content[start:end + 1])

    def _request_batch(self, indexed_texts):
        if not self.api_key:
            raise RuntimeError("9Router API key is not configured")
        items = [{"id": index, "text": text} for index, text in indexed_texts]
        system_prompt = (
            "You are a precise video graphics translator. Translate each input into "
            f"{self.target_language}. Preserve numbers, units, formulas, product names, "
            "and concise on-screen meaning. Return ONLY valid JSON in this exact shape: "
            '{"translations":[{"id":0,"translation":"..."}]}. Do not add markdown.'
        )
        payload = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                parsed = self._extract_json(content)
                translations = parsed.get("translations", [])
                return {
                    int(item["id"]): str(item["translation"]).strip()
                    for item in translations
                    if isinstance(item, dict) and "id" in item and item.get("translation")
                }
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"9Router translation request failed: {last_error}") from last_error

    def translate_many(self, texts, batch_size=24):
        if self.model.casefold() == "auto":
            models = self.list_models()
            if not models:
                raise RuntimeError("9Router has no available model")
            self.model = self.find_working_translation_model(models)
        unique_texts = list(dict.fromkeys(text.strip() for text in texts if text and text.strip()))
        results = {}
        missing = []
        with self._cache_lock:
            for text in unique_texts:
                cached = self._cache.get(self._cache_key(text))
                if cached:
                    results[text] = cached
                else:
                    missing.append(text)

        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            translated = self._request_batch(list(enumerate(batch)))
            absent = []
            for index, text in enumerate(batch):
                value = translated.get(index)
                if value:
                    results[text] = value
                else:
                    absent.append(text)
            # Some models occasionally omit an item from a larger JSON batch.
            # Retry omitted strings individually rather than silently dropping them.
            for text in absent:
                single = self._request_batch([(0, text)]).get(0)
                if not single:
                    raise RuntimeError(f"9Router returned no translation for: {text}")
                results[text] = single

        if missing:
            with self._cache_lock:
                for text in missing:
                    self._cache[self._cache_key(text)] = results[text]
                self._save_cache()
        return results

    def _fetch_models(self, timeout):
        response = self.session.get(
            f"{self.api_v1_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            item.get("id") for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    @property
    def dashboard_url(self):
        parsed = urlparse(self.api_v1_url)
        return f"{parsed.scheme}://{parsed.netloc}/dashboard"

    @staticmethod
    def _request_error_summary(error):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        return f"HTTP {status}" if status is not None else type(error).__name__

    def _probe_model(self, model):
        response = self.session.post(
            self.chat_completions_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "stream": False,
                "temperature": 0,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Reply only OK"}],
            },
            timeout=min(self.timeout, 30),
        )
        response.raise_for_status()

    def find_working_translation_model(self, models):
        """Probe one preferred model per provider before expensive video work."""
        models = list(dict.fromkeys(model for model in models or [] if model))
        if self.model.casefold() != "auto":
            candidates = [self.model]
        else:
            preferred = self.choose_translation_model(models)
            candidates = ([preferred] if preferred else []) + models

        provider_candidates = []
        seen_providers = set()
        for model in candidates:
            provider = model.split("/", 1)[0].casefold()
            if provider in seen_providers:
                continue
            seen_providers.add(provider)
            provider_candidates.append(model)

        failures = []
        for model in provider_candidates:
            try:
                self._probe_model(model)
                return model
            except requests.RequestException as error:
                failures.append(f"{model}: {self._request_error_summary(error)}")

        detail = "; ".join(failures) or "no model returned by /v1/models"
        raise RuntimeError(
            "9Router is running, but no connected provider can process a "
            f"translation request ({detail}). Reconnect a provider at "
            f"{self.dashboard_url}, then use Test in Advanced Settings."
        )

    def _is_local_endpoint(self):
        return urlparse(self.api_v1_url).hostname in {
            "127.0.0.1", "localhost", "::1"
        }

    @classmethod
    def _start_local_router(cls):
        """Start an installed local 9Router without opening another window."""
        with cls._local_start_lock:
            command = shutil.which("9router.cmd") or shutil.which("9router")
            if not command:
                return False
            arguments = [command, "--no-browser", "--skip-update"]
            creation_flags = 0
            if os.name == "nt":
                # npm exposes global CLIs as .cmd launchers on Windows.
                if command.lower().endswith((".cmd", ".bat")):
                    arguments = [
                        os.environ.get("COMSPEC", "cmd.exe"),
                        "/d", "/c", command,
                        "--no-browser", "--skip-update",
                    ]
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                subprocess.Popen(
                    arguments,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    creationflags=creation_flags,
                )
                return True
            except OSError:
                return False

    def list_models(self):
        if not self.api_key:
            raise RuntimeError("9Router API key is not configured")
        try:
            return self._fetch_models(timeout=5)
        except requests.ConnectionError as first_error:
            if not self._is_local_endpoint() or not self._start_local_router():
                raise RuntimeError(
                    f"Cannot reach 9Router at {self.api_v1_url}. Start 9Router "
                    "or verify the API endpoint in Advanced Settings."
                ) from first_error

            # A cold 9Router start can take several seconds. Retry only the
            # connection failure; HTTP authentication errors should surface.
            deadline = time.monotonic() + 20
            last_error = first_error
            while time.monotonic() < deadline:
                time.sleep(0.5)
                try:
                    return self._fetch_models(timeout=3)
                except requests.ConnectionError as error:
                    last_error = error
            raise RuntimeError(
                f"9Router was started but did not become ready at "
                f"{self.api_v1_url}. Open the 9Router dashboard and try again."
            ) from last_error
        except requests.Timeout as error:
            raise RuntimeError(
                f"9Router timed out at {self.api_v1_url}. Restart 9Router and try again."
            ) from error
        except requests.HTTPError as error:
            status = getattr(error.response, "status_code", "unknown")
            raise RuntimeError(
                f"9Router rejected the request (HTTP {status}). Check the API key "
                "and connected provider in the 9Router dashboard."
            ) from error
        except (ValueError, TypeError, AttributeError) as error:
            raise RuntimeError(
                "9Router returned an invalid model list. Restart 9Router and try again."
            ) from error


class TranslationPlan:
    def __init__(self, tracks):
        self.tracks = tracks
        self._font_path = self._find_font()

    @staticmethod
    def _find_font():
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        return next((path for path in candidates if os.path.exists(path)), None)

    def _font(self, size):
        if self._font_path:
            return ImageFont.truetype(self._font_path, size=size)
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(draw, text, font, max_width):
        words = text.split()
        if not words:
            return [text]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def render(self, frame, frame_no):
        active = [
            track for track in self.tracks
            if track.translated_text and track.start_frame <= frame_no <= track.end_frame
        ]
        if not active:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        frame_width, frame_height = base.size

        for track in active:
            xmin, xmax, ymin, ymax = track.box_at(frame_no)
            source_width = max(40, xmax - xmin)
            source_height = max(18, ymax - ymin)
            max_width = min(frame_width - 16, max(180, round(source_width * 2.5)))
            font_size = max(16, min(48, round(source_height * 0.9)))
            font = self._font(font_size)
            lines = self._wrap_text(draw, track.translated_text, font, max_width - 20)
            line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=1) for line in lines]
            text_width = min(max_width, max(box[2] - box[0] for box in line_boxes) + 20)
            line_height = max(box[3] - box[1] for box in line_boxes) + 4
            text_height = line_height * len(lines) + 12
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            left = max(8, min(round(center_x - text_width / 2), frame_width - text_width - 8))
            top = max(8, min(round(center_y - text_height / 2), frame_height - text_height - 8))
            right = left + text_width
            bottom = top + text_height
            draw.rounded_rectangle((left, top, right, bottom), radius=6, fill=(0, 0, 0, 175))
            for line_index, line in enumerate(lines):
                box = line_boxes[line_index]
                width = box[2] - box[0]
                x = left + (text_width - width) / 2
                y = top + 6 + line_index * line_height
                draw.text(
                    (x, y), line, font=font, fill=(255, 255, 255, 255),
                    stroke_width=1, stroke_fill=(0, 0, 0, 255),
                )

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)

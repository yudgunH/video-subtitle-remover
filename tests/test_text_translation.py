import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

import numpy as np

from backend.tools.text_translation import (
    NineRouterTranslator,
    TranslationPlan,
    TranslationTrack,
    build_translation_tracks,
)


class TranslationTrackTests(unittest.TestCase):
    def test_tracks_text_and_excludes_selected_subtitle_zone(self):
        records = {
            10: [
                {"box": (50, 170, 40, 75), "text": "带宽", "score": 0.92},
                {"box": (80, 220, 620, 670), "text": "中文字幕", "score": 0.95},
            ],
            12: [
                {"box": (54, 174, 42, 77), "text": "带宽", "score": 0.96},
                {"box": (84, 224, 620, 670), "text": "中文字幕", "score": 0.94},
            ],
        }

        tracks = build_translation_tracks(
            records,
            exclusion_areas=[(580, 720, 0, 1280)],
            sample_step=2,
            total_frames=100,
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].source_text, "带宽")
        self.assertEqual(tracks[0].start_frame, 9)
        self.assertEqual(tracks[0].end_frame, 14)
        self.assertEqual(tracks[0].box_at(11), (52, 172, 41, 76))

    def test_renderer_only_changes_active_frames(self):
        track = TranslationTrack(
            "带宽", 5, 10,
            keyframes=[(5, (30, 130, 20, 50))],
            translated_text="Băng thông",
        )
        plan = TranslationPlan([track])
        frame = np.full((100, 240, 3), 80, dtype=np.uint8)

        self.assertTrue(np.array_equal(plan.render(frame, 4), frame))
        self.assertFalse(np.array_equal(plan.render(frame, 7), frame))


class NineRouterTranslatorTests(unittest.TestCase):
    def test_endpoint_normalization(self):
        translator = NineRouterTranslator(
            "http://localhost:20128/v1", "key", "auto", "Vietnamese"
        )
        self.assertEqual(
            translator.chat_completions_url,
            "http://localhost:20128/v1/chat/completions",
        )

    def test_auto_model_prefers_fast_general_translation_model(self):
        models = [
            "xai/grok-4",
            "xai/grok-4-fast-reasoning",
            "kr/claude-opus-4.8",
            "kr/claude-haiku-4.5",
        ]
        self.assertEqual(
            NineRouterTranslator.choose_translation_model(models),
            "kr/claude-haiku-4.5",
        )

    def test_model_probe_falls_back_to_another_provider(self):
        translator = NineRouterTranslator(
            "http://127.0.0.1:20128/v1", "secret", "auto", "Vietnamese"
        )
        forbidden_response = Mock(status_code=403)
        forbidden = requests.HTTPError(response=forbidden_response)
        with patch.object(
            translator,
            "_probe_model",
            side_effect=[forbidden, None],
        ) as probe:
            selected = translator.find_working_translation_model([
                "kr/claude-haiku-4.5",
                "kr/glm-5",
                "xai/grok-4",
            ])

        self.assertEqual(selected, "xai/grok-4")
        self.assertEqual(probe.call_count, 2)

    def test_model_probe_reports_provider_reconnect_action(self):
        translator = NineRouterTranslator(
            "http://127.0.0.1:20128/v1", "secret", "auto", "Vietnamese"
        )
        forbidden = requests.HTTPError(response=Mock(status_code=403))
        with patch.object(translator, "_probe_model", side_effect=forbidden):
            with self.assertRaisesRegex(RuntimeError, "Reconnect a provider"):
                translator.find_working_translation_model([
                    "kr/claude-haiku-4.5",
                    "xai/grok-4",
                ])

    def test_batch_translation_is_cached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            translator = NineRouterTranslator(
                "http://localhost:20128/v1", "secret", "kr/glm-5", "Vietnamese"
            )
            translator._cache_path = os.path.join(temporary_directory, "cache.json")
            translator._cache = {}

            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": '{"translations":[{"id":0,"translation":"Băng thông"}]}'
                    }
                }]
            }
            translator.session = Mock()
            translator.session.post.return_value = response

            first = translator.translate_many(["带宽", "带宽"])
            second = translator.translate_many(["带宽"])

            self.assertEqual(first["带宽"], "Băng thông")
            self.assertEqual(second["带宽"], "Băng thông")
            self.assertEqual(translator.session.post.call_count, 1)
            self.assertTrue(os.path.exists(translator._cache_path))

    def test_local_router_is_started_after_connection_refusal(self):
        translator = NineRouterTranslator(
            "http://127.0.0.1:20128/v1", "secret", "auto", "Vietnamese"
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"id": "provider/model"}]}
        translator.session = Mock()
        translator.session.get.side_effect = [
            requests.ConnectionError("connection refused"),
            response,
        ]

        with patch.object(
            translator, "_start_local_router", return_value=True
        ) as start_router, patch(
            "backend.tools.text_translation.time.sleep", return_value=None
        ):
            models = translator.list_models()

        self.assertEqual(models, ["provider/model"])
        start_router.assert_called_once()

    def test_remote_router_failure_has_actionable_message(self):
        translator = NineRouterTranslator(
            "https://router.invalid/v1", "secret", "auto", "Vietnamese"
        )
        translator.session = Mock()
        translator.session.get.side_effect = requests.ConnectionError(
            "connection refused"
        )

        with self.assertRaisesRegex(RuntimeError, "Cannot reach 9Router"):
            translator.list_models()


if __name__ == "__main__":
    unittest.main()

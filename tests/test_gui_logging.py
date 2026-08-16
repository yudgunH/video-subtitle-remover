import unittest
from unittest.mock import Mock, patch

from ui.home_interface import (
    HomeInterface,
    _is_full_frame_selection,
    _parse_normalized_areas,
)
from backend.tools.subtitle_remover_remote_call import Command, SubtitleRemoverRemoteCall


class GuiLoggingTests(unittest.TestCase):
    def test_full_frame_is_not_a_valid_caption_exclusion(self):
        self.assertTrue(_is_full_frame_selection([(0, 1, 0, 1)]))
        self.assertFalse(_is_full_frame_selection([(0.82, 0.99, 0.05, 0.95)]))

    def test_parses_persisted_caption_exclusion(self):
        self.assertEqual(
            _parse_normalized_areas("0.82,0.99,0.05,0.95"),
            [(0.82, 0.99, 0.05, 0.95)],
        )

    def test_cjk_console_encoding_error_does_not_interrupt_ui_log(self):
        home = Mock()
        home.output_text = Mock()
        home.auto_scroll = False
        encoding_error = UnicodeEncodeError(
            "cp1252", "中文文件.mp4", 0, 1, "character maps to undefined"
        )

        with patch("builtins.print", side_effect=encoding_error):
            HomeInterface.append_output(home, "Opened 中文文件.mp4")

        home.output_text.append.assert_called_once()

    def test_worker_sends_stable_error_text_to_gui(self):
        queue = Mock()
        error = RuntimeError("Cannot reach 9Router at http://127.0.0.1:20128/v1")

        SubtitleRemoverRemoteCall.remote_call_catch_error(queue, error)

        command, arguments = queue.put.call_args.args[0]
        self.assertEqual(command, Command.ERROR)
        self.assertEqual(arguments, (str(error),))


if __name__ == "__main__":
    unittest.main()

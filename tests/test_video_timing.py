import unittest

import numpy as np

from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.video_io import FFmpegVideoWriter


class VideoTimingTests(unittest.TestCase):
    def test_framecrc_parser_recovers_vfr_cadence_and_timestamps(self):
        framecrc = """#tb 0: 1/1000
0, 0, 0, 40, 100, 0x0
0, 40, 40, 40, 100, 0x0
0, 80, 120, 40, 100, 0x0
"""

        timing = FFmpegCLI._parse_framecrc_timing(
            framecrc, expected_frame_count=3
        )

        self.assertIsNotNone(timing)
        self.assertEqual(timing.nominal_fps, 25.0)
        self.assertEqual(timing.timestamps, (0.0, 0.04, 0.12))
        self.assertAlmostEqual(timing.duration, 0.16)
        self.assertTrue(timing.variable_frame_rate)

    def test_writer_repeats_previous_processed_frame_for_pts_gap(self):
        writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
        writer.fps = 25.0
        writer.source_timestamps = (0.0, 0.08, 0.12)
        writer.source_duration = None
        writer._source_frames_received = 0
        writer._encoded_frames_written = 0
        writer._last_frame = None
        emitted = []

        def capture(frame):
            emitted.append(int(frame[0, 0, 0]))
            writer._encoded_frames_written += 1

        writer._write_encoded_frame = capture
        for value in (1, 2, 3):
            writer.write(np.full((2, 2, 3), value, dtype=np.uint8))

        self.assertEqual(emitted, [1, 1, 2, 3])
        self.assertEqual(writer._source_frames_received, 3)


if __name__ == "__main__":
    unittest.main()

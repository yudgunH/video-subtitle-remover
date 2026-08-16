import os
import stat
import subprocess
import math
import re
from dataclasses import dataclass
from functools import reduce

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR


@dataclass(frozen=True)
class VideoFrameTiming:
    timestamps: tuple
    nominal_fps: float
    duration: float
    variable_frame_rate: bool

class FFmpegCLI:
    
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance
    
    def __init__(self):
        self._nvenc_supported = None
        os.chmod(self.ffmpeg_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)

    def supports_h264_nvenc(self):
        """Run a one-frame encoder smoke test and cache the result."""
        if self._nvenc_supported is not None:
            return self._nvenc_supported

        command = [
            self.ffmpeg_path,
            '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '256x256', '-r', '1',
            '-i', '-', '-frames:v', '1',
            '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '18', '-b:v', '0',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(
                command,
                input=bytes(256 * 256 * 3),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            self._nvenc_supported = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            self._nvenc_supported = False
        return self._nvenc_supported

    @staticmethod
    def _parse_framecrc_timing(output, expected_frame_count=None):
        """Parse packet PTS without decoding the video stream."""
        time_base = None
        packets = []
        for line in (output or "").splitlines():
            match = re.match(r"#tb\s+0:\s*(\d+)\s*/\s*(\d+)", line)
            if match:
                numerator, denominator = (int(value) for value in match.groups())
                if numerator > 0 and denominator > 0:
                    time_base = numerator / denominator
                continue
            if not line.startswith("0,"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4:
                continue
            try:
                pts = int(fields[2])
                duration = max(1, int(fields[3]))
            except ValueError:
                continue
            packets.append((pts, duration))

        if time_base is None or not packets:
            return None
        packets.sort(key=lambda item: item[0])
        if expected_frame_count and len(packets) != int(expected_frame_count):
            return None

        first_pts = packets[0][0]
        timestamps = tuple((pts - first_pts) * time_base for pts, _ in packets)
        delta_units = [
            packets[index][0] - packets[index - 1][0]
            for index in range(1, len(packets))
            if packets[index][0] > packets[index - 1][0]
        ]
        duration_units = [duration for _, duration in packets if duration > 0]
        cadence_units = reduce(math.gcd, delta_units + duration_units)
        nominal_fps = 1.0 / (cadence_units * time_base)
        if not 1.0 <= nominal_fps <= 240.0:
            return None
        end_pts = max(pts + duration for pts, duration in packets)
        total_duration = (end_pts - first_pts) * time_base
        return VideoFrameTiming(
            timestamps=timestamps,
            nominal_fps=nominal_fps,
            duration=total_duration,
            variable_frame_rate=len(set(delta_units)) > 1,
        )

    def probe_video_frame_timing(self, video_path, expected_frame_count=None):
        """Read per-frame presentation timing through FFmpeg's framecrc muxer."""
        command = [
            self.ffmpeg_path,
            '-hide_banner', '-loglevel', 'error',
            '-i', str(video_path),
            '-map', '0:v:0', '-c', 'copy', '-f', 'framecrc', '-'
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return self._parse_framecrc_timing(
            result.stdout, expected_frame_count=expected_frame_count
        )
        
    @property
    def ffmpeg_path(self):
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')
            merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(BASE_DIR, 'ffmpeg',  'linux_x64', 'ffmpeg')
        else:
            return os.path.join(BASE_DIR, 'ffmpeg', 'macos', 'ffmpeg')

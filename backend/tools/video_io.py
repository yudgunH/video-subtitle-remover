import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI
from .hardware_accelerator import HardwareAccelerator


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        return self._buffer.get()

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧；CUDA可用时优先NVENC，否则使用libx264。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    def __init__(
        self,
        output_path,
        fps,
        size,
        source_timestamps=None,
        source_duration=None,
    ):
        w, h = size
        ffmpeg_cli = FFmpegCLI.instance()
        use_nvenc = (
            HardwareAccelerator.instance().has_cuda()
            and ffmpeg_cli.supports_h264_nvenc()
        )
        encoder_args = (
            ['-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '18', '-b:v', '0']
            if use_nvenc
            else ['-c:v', 'libx264', '-crf', '18', '-preset', 'fast']
        )
        self.encoder_name = 'h264_nvenc' if use_nvenc else 'libx264'
        self.fps = float(fps)
        self.source_timestamps = tuple(source_timestamps or ())
        self.source_duration = (
            float(source_duration) if source_duration is not None else None
        )
        self._source_frames_received = 0
        self._encoded_frames_written = 0
        self._last_frame = None
        print(f"FFmpeg video encoder: {self.encoder_name}")
        cmd = [
            ffmpeg_cli.ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            *encoder_args,
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _write_encoded_frame(self, frame):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
        self._encoded_frames_written += 1

    def write(self, frame):
        """写入处理帧，并用重复帧保留源视频的 VFR 时间戳间隔。"""
        timestamp = None
        if self._source_frames_received < len(self.source_timestamps):
            timestamp = self.source_timestamps[self._source_frames_received]
        self._source_frames_received += 1

        if timestamp is not None:
            target_index = max(
                self._encoded_frames_written,
                int(round(timestamp * self.fps)),
            )
            if self._last_frame is not None:
                while self._encoded_frames_written < target_index:
                    self._write_encoded_frame(self._last_frame)

        self._write_encoded_frame(frame)
        self._last_frame = frame.copy()

    def release(self):
        """关闭管道并等待编码完成。"""
        if self._last_frame is not None and self.source_duration is not None:
            expected_frames = max(
                self._encoded_frames_written,
                int(round(self.source_duration * self.fps)),
            )
            while self._encoded_frames_written < expected_frames:
                self._write_encoded_frame(self._last_frame)
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)

import os
import stat
import subprocess

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR

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

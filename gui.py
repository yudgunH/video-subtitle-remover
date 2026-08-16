# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao（原作者） / 改写：Jason Eric
@Time    : 2023/4/1 6:07 下午（原始时间）
@FileName: gui.py
@desc: 字幕去除器图形化界面（由 PySimpleGUI 改写为 PySide6）
"""

import sys
import os
import configparser

# This must run before importing OpenCV/Paddle/Torch/Qt so their caches and
# temporary files are created in the user-selected portable data directory.
from backend.tools.app_paths import initialize_runtime_environment, resource_root

initialize_runtime_environment()

import cv2
import multiprocessing
from PySide6.QtCore import Qt, QTranslator
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QLabel
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, setThemeColor, OptionsConfigItem,
                          OptionsValidator, SubtitleLabel, HollowHandleStyle, qconfig, ConfigItem, QConfig,
                          NavigationWidget, NavigationItemPosition, isDarkTheme, InfoBar)

from qframelesswindow.utils import getSystemAccentColor
from backend.config import config, tr, VERSION
from backend.tools.theme_listener import SystemThemeListener
from backend.tools.process_manager import ProcessManager
from ui.advanced_setting_interface import AdvancedSettingInterface
from ui.home_interface import HomeInterface


def run_package_self_test():
    """Load bundled OCR models and verify packaged runtime dependencies."""

    import json
    import subprocess
    import traceback

    from backend.tools.app_paths import get_data_path
    from backend.tools.ffmpeg_cli import FFmpegCLI
    from backend.tools.model_config import ModelConfig

    report_path = get_data_path("package_self_test.json", create_parent=True)
    report = {"success": False, "version": VERSION}
    try:
        from paddleocr import TextDetection, TextRecognition

        model_config = ModelConfig()
        detector = TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="cpu",
            enable_hpi=False,
        )
        recognizer = TextRecognition(
            model_name=model_config.REC_MODEL_NAME,
            model_dir=model_config.REC_MODEL_DIR,
            device="cpu",
            enable_hpi=False,
        )
        ffmpeg_result = subprocess.run(
            [FFmpegCLI.instance().ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if ffmpeg_result.returncode != 0:
            raise RuntimeError("Bundled FFmpeg failed its version check")
        report.update(
            success=True,
            detection_model=model_config.DET_MODEL_NAME,
            recognition_model=model_config.REC_MODEL_NAME,
            ffmpeg=ffmpeg_result.stdout.splitlines()[0],
        )
        del detector, recognizer
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report["success"] else 1


class SubtitleExtractorGUI(FluentWindow): 
    def __init__(self):
        super().__init__()
        # 禁用云母效果
        self.setMicaEffectEnabled(False)
        # 设置深色主题并跟随系统主题色
        # setTheme(Theme.LIGHT)
        # setThemeColor(getSystemAccentColor(), save=True)

        # 初始化系统主题监听器并连接信号
        # self.themeListener = SystemThemeListener(self)
        # self.themeListener.start()
 
        # 设置窗口图标
        self.setWindowIcon(QtGui.QIcon(str(resource_root() / "design" / "vsr.ico")))
        self.setWindowTitle(tr['SubtitleExtractorGUI']['Title'] + " v" + VERSION)
        # 创建界面布局
        self._create_layout()
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(
            min(900, max(1, available.width() - 24)),
            min(620, max(1, available.height() - 24)),
        )
        self._connectSignalToSlot()
        self._lazy_check_update()

    def _lazy_check_update(self):
        """ 延迟检查更新 """
        if not config.checkUpdateOnStartup.value:
            return
        self.check_update_timer = QtCore.QTimer(self)
        self.check_update_timer.setSingleShot(True)
        self.check_update_timer.timeout.connect(lambda: self.advancedSettingInterface.check_update(ignore=True))
        self.check_update_timer.start(2000)

    def _connectSignalToSlot(self):
        config.appRestartSig.connect(self._showRestartTooltip)

    def _showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            'Updated successfully',
            'Configuration takes effect after restart',
            duration=5000,
            parent=self
        )

    def _create_layout(self):
        # 创建主页面和高级设置页面
        self.homeInterface = HomeInterface(self)
        self.homeInterface.setObjectName("HomeInterface")
        self.advancedSettingInterface = AdvancedSettingInterface(self)
        self.advancedSettingInterface.setObjectName("AdvancedSettingInterface")
        
        # 添加到主窗口作为子界面
        self.addSubInterface(self.homeInterface,FluentIcon.HOME, tr['SubtitleExtractorGUI']['Title'])
        self.addSubInterface(self.advancedSettingInterface, FluentIcon.SETTING, tr['Setting']['AdvancedSetting'], NavigationItemPosition.BOTTOM)

    def on_navigation_item_changed(self, key):
        """导航项变更时的处理函数"""
        if key == 'main':
            self.stackWidget.setCurrentIndex(0)
        elif key == 'advanced':
            self.stackWidget.setCurrentIndex(1)

    def closeEvent(self, event):
        """程序关闭时保存窗口位置并清理资源"""
        self.save_window_position()
        ProcessManager.instance().terminate_all()
        super().closeEvent(event)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

    def save_window_position(self):
        """保存窗口位置到配置文件"""
        # 保存窗口位置和大小
        config.set(config.windowX, self.x())
        config.set(config.windowY, self.y())
        config.set(config.windowW, self.width())
        config.set(config.windowH, self.height())

    def update_progress(self):
        # 定时器轮询更新进度（现在更新到视频滑块上）
        if self.se is not None:
            try:
                pos = min(self.frame_count - 1, int(self.se.progress_total / 100 * self.frame_count))
                if pos != self.video_slider.value():
                    self.video_slider.setValue(pos)
                # 检查是否完成
                if self.se.isFinished:
                    self.processing_finished()
            except Exception as e:
                # 捕获任何异常，防止崩溃
                print(f"更新进度时出错: {str(e)}")

    def load_window_position(self):
        """Restore a window geometry that always fits the usable desktop."""
        try:
            x = config.windowX.value
            y = config.windowY.value
            screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
            if screen is None:
                screen = QtWidgets.QApplication.primaryScreen()
            screen_rect = screen.availableGeometry()

            # Leave a small safety margin for the frame/shadow. Old versions
            # saved 1200 px high windows, which could extend below the taskbar.
            max_width = max(1, screen_rect.width() - 24)
            max_height = max(1, screen_rect.height() - 24)
            min_width = min(900, max_width)
            min_height = min(620, max_height)
            width = max(min_width, min(int(config.windowW.value or 1280), max_width))
            height = max(min_height, min(int(config.windowH.value or 820), max_height))

            if x is None or y is None:
                x = screen_rect.left() + (screen_rect.width() - width) // 2
                y = screen_rect.top() + (screen_rect.height() - height) // 2
            else:
                x = max(screen_rect.left(), min(int(x), screen_rect.right() - width + 1))
                y = max(screen_rect.top(), min(int(y), screen_rect.bottom() - height + 1))

            self.setGeometry(x, y, width, height)
        except Exception as e:
            print(e)
            self.center_window()
    
    def center_window(self):
        """将窗口居中显示"""
        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(max(self.width(), 900), max(1, screen_rect.width() - 24)),
            min(max(self.height(), 620), max(1, screen_rect.height() - 24)),
        )
        window_rect = self.frameGeometry()
        center_point = screen_rect.center()
        window_rect.moveCenter(center_point)
        self.move(window_rect.topLeft())

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 检测Ctrl+C组合键
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            print("\n程序被用户中断(Ctrl+C)，正在退出...")
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    if "--package-self-test" in sys.argv:
        raise SystemExit(run_package_self_test())
    multiprocessing.set_start_method("spawn")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    window = SubtitleExtractorGUI()
    # 先设置透明, 再显示, 否则会有闪烁的效果
    window.setWindowOpacity(0.0)
    window.show()
    window.load_window_position()
    # 使用动画效果逐渐显示窗口
    animation = QtCore.QPropertyAnimation(window, b"windowOpacity")
    animation.setDuration(300)  # 300毫秒的动画
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.start()
    app.exec()

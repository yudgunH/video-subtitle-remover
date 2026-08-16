from PySide6 import QtWidgets
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard)
from backend.config import (
    config,
    tr,
    HARDWARD_ACCELERATION_OPTION,
    TRANSLATION_LANGUAGE_OPTIONS,
)
from backend.tools.constant import InpaintMode, SubtitleDetectMode

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        
        # 界面语言设置
        self.interface_combo = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["SubtitleExtractorGUI"]["InterfaceLanguage"],
            content="",
            parent=parent,
            texts=config.intefaceTexts.keys(),
        )
        self.addWidget(self.interface_combo)
        
        self._inpaint_options = list(config.inpaintMode.validator.options)

        # 处理模式设置
        self.inpaint_mode_combo = ComboBoxSettingCard(
            configItem=config.inpaintMode,
            icon=FluentIcon.GLOBE,
            title=tr["SubtitleExtractorGUI"]["InpaintMode"],
            content=self._get_inpaint_mode_description(config.inpaintMode.value),
            parent=parent,
            texts=[list(tr['InpaintMode'].values())[i] for i, _ in enumerate(self._inpaint_options)],
        )
        self.inpaint_mode_combo.comboBox.currentIndexChanged.connect(self._on_inpaint_mode_changed)
        self.addWidget(self.inpaint_mode_combo)

        self.subtitle_detect_model_combo = ComboBoxSettingCard(
            configItem=config.subtitleDetectMode,
            icon=FluentIcon.SEARCH,
            title=tr["SubtitleExtractorGUI"]["SubtitleDetectMode"],
            content=tr["Setting"]["SubtitleDetectModeDesc"],
            parent=parent,
            texts=[list(tr['SubtitleDetectMode'].values())[i] for i,_ in enumerate(config.subtitleDetectMode.validator.options)],
        )
        self.addWidget(self.subtitle_detect_model_combo)

        self.remove_cjk_text = SwitchSettingCard(
            configItem=config.removeCjkText,
            icon=FluentIcon.LANGUAGE,
            title=tr["Setting"]["RemoveCjkText"],
            content=tr["Setting"]["RemoveCjkTextDesc"],
            parent=parent
        )
        self.addWidget(self.remove_cjk_text)

        self.translate_non_subtitle_cjk = SwitchSettingCard(
            configItem=config.translateNonSubtitleCjk,
            icon=FluentIcon.MESSAGE,
            title=tr["Setting"]["TranslateNonSubtitleCjk"],
            content=tr["Setting"]["TranslateNonSubtitleCjkDesc"],
            parent=parent,
        )
        self.translate_non_subtitle_cjk.checkedChanged.connect(
            self._on_translation_enabled_changed
        )
        self.addWidget(self.translate_non_subtitle_cjk)

        self.translation_target_language = ComboBoxSettingCard(
            configItem=config.translationTargetLanguage,
            icon=FluentIcon.LANGUAGE,
            title=tr["Setting"]["TranslationTargetLanguage"],
            content=tr["Setting"]["TranslationTargetLanguageDesc"],
            parent=parent,
            texts=list(TRANSLATION_LANGUAGE_OPTIONS.keys()),
        )
        self.translation_target_language.comboBox.setEnabled(
            config.translateNonSubtitleCjk.value
        )
        self.addWidget(self.translation_target_language)

        # 是否启用硬件加速
        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=parent
        )
        self.addWidget(self.hardware_acceleration)
        # 如果硬件加速选项被禁用, 设置硬件加速为False并只读
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
            config.set(config.hardwareAcceleration, False)
        # 添加一些空间
        self.addStretch(1)

    @staticmethod
    def _get_inpaint_mode_description(mode):
        return tr["InpaintModeDescription"].get(
            mode.name,
            tr["SubtitleExtractorGUI"]["InpaintModeDesc"],
        )

    def _on_inpaint_mode_changed(self, index):
        if 0 <= index < len(self._inpaint_options):
            self.inpaint_mode_combo.setContent(
                self._get_inpaint_mode_description(self._inpaint_options[index])
            )

    def _on_translation_enabled_changed(self, enabled):
        self.translation_target_language.comboBox.setEnabled(enabled)
        if enabled and not config.removeCjkText.value:
            config.set(config.removeCjkText, True)
    
    def set_inpaint_mode_enabled(self, enabled):
        """启用或禁用 inpaint 模式下拉框"""
        self.inpaint_mode_combo.comboBox.setEnabled(enabled)

    def reset_setting(self):
        """重置所有设置为默认值"""
        # 这里需要实现重置逻辑
        pass

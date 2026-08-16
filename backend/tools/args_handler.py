import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    parser.add_argument(
        "--remove-chinese-text", "--remove-cjk-text",
        dest="remove_cjk_text", action="store_true",
        help=(
            "Remove all detected text inside -c/--subtitle-area-coords and "
            "only confirmed Chinese text elsewhere"
        )
    )
    parser.add_argument(
        "--translate-non-subtitle-chinese", "--translate-non-subtitle-cjk",
        dest="translate_non_subtitle_cjk", action="store_true",
        help=(
            "Translate confirmed Chinese text outside -c/--subtitle-area-coords with 9Router. "
            "Set the secret in VSR_9ROUTER_API_KEY."
        ),
    )
    parser.add_argument(
        "--translation-target-language", default="Vietnamese",
        choices=[
            "Vietnamese", "English", "Simplified Chinese", "Japanese",
            "Korean", "Spanish",
        ],
        help="Target language for translated on-screen text",
    )
    parser.add_argument(
        "--nine-router-base-url", default="https://9router.hbfstudio.site/v1",
        help="9Router OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--nine-router-model", default="auto",
        help="Model identifier routed by 9Router",
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    return args

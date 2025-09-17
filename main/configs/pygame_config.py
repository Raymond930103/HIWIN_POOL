"""
Pygame UI configuration

Edit these values to customize visualization behavior across:
- main/gui/simulator.py
- main/gui/visualize.py
- main/cli/shot_cli.py
- webapp/render.py (offscreen renderer)

Controls include toggling lines/pockets/grid and ball colors,
including optional highlighting of the target ball when a plan exists.
"""

from dataclasses import dataclass
from typing import Tuple


Color = Tuple[int, int, int]


@dataclass
class PygameUIConfig:
    # Feature toggles
    SHOW_SOLID_LINES: bool = False            # red/yellow solid lines
    SHOW_DASHED_GUIDES: bool = False          # gray dashed guide segments
    SHOW_POCKETS: bool = False               # draw pocket circles
    SHOW_GRID: bool = True                  # draw metric grid/labels (simulator/visualizer/webapp)

    # Highlighting
    HIGHLIGHT_TARGET_ON_PLAN: bool = False   # change target ball color if a plan exists
    TARGET_AS_NORMAL: bool = True          # force target ball to use normal ball color

    # Colors
    COLOR_TABLE: Color = (18, 95, 29)
    COLOR_RAIL: Color = (60, 30, 10)
    COLOR_POCKET: Color = (25, 12, 4)

    COLOR_CUE_BALL: Color = (245, 245, 245)
    COLOR_TARGET_BALL: Color = (255, 90, 40)
    COLOR_TARGET_HIGHLIGHT: Color = (255, 200, 0)  # bright yellow/orange
    COLOR_OTHER_BALLS: Color = (40, 120, 255)

    COLOR_LINE_PRIMARY: Color = (250, 0, 0)        # cue path
    COLOR_LINE_SECONDARY: Color = (255, 255, 0)    # target to pocket
    COLOR_DASHED: Color = (185, 185, 185)


# Singleton-style config object used by all pygame modules
CONFIG = PygameUIConfig()

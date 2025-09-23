import os
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Offscreen for pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

# Import simulator constants
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.gui import simulator as sim
from main.configs.pygame_config import CONFIG as PGCFG


def _render_surface(cue: Tuple[float, float], target: Tuple[float, float], blockers: List[Tuple[float, float]],
                    info: Optional[dict], labels: Optional[List[Tuple[str, Tuple[float, float]]]] = None) -> pygame.Surface:
    pygame.init()
    sim.LABEL_FONT = pygame.font.SysFont(None, 18)

    w, h = sim.TABLE
    width = int(w * sim.SCALE + sim.MARGIN * 2)
    height = int(h * sim.SCALE + sim.MARGIN * 2)
    surface = pygame.Surface((width, height))

    # pockets
    pkts = [np.array([0, 0]), np.array([w/2, 0]), np.array([w, 0]),
            np.array([0, h]), np.array([w/2, h]), np.array([w, h])]

    # background
    surface.fill(sim.RAIL)
    pygame.draw.rect(surface, sim.GREEN, (sim.MARGIN, sim.MARGIN, w*sim.SCALE, h*sim.SCALE))
    if PGCFG.SHOW_GRID:
        sim.draw_grid(surface, w, h)

    # pockets
    if PGCFG.SHOW_POCKETS:
        for pk in pkts:
            pygame.draw.circle(surface, sim.PKCOL, sim.px(pk), sim.R_PK)

    # balls
    draw_ball = lambda p, c: pygame.draw.circle(surface, c, sim.px(p), sim.R_BALL)
    draw_ball(cue, sim.CUE)
    if PGCFG.TARGET_AS_NORMAL:
        tgt_col = sim.OTH
    elif PGCFG.HIGHLIGHT_TARGET_ON_PLAN and info:
        tgt_col = PGCFG.COLOR_TARGET_HIGHLIGHT
    else:
        tgt_col = sim.TARGET
    draw_ball(target, tgt_col)
    for b in blockers:
        draw_ball(b, sim.OTH)

    # labels (ball numbers)
    if labels:
        font = pygame.font.SysFont(None, 24)
        for text, pos in labels:
            tx = font.render(str(text), True, (255, 255, 255))
            # simple outline for readability
            ox, oy = sim.px(pos)
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                surface.blit(font.render(str(text), True, (0, 0, 0)), (ox + dx - tx.get_width()//2, oy + dy - tx.get_height()//2))
            surface.blit(tx, (ox - tx.get_width()//2, oy - tx.get_height()//2))

    # path
    if info:
        G = np.array(info["ghost"]) if "ghost" in info else None
        if G is not None:
            if info.get("type") == "direct":
                if PGCFG.SHOW_DASHED_GUIDES:
                    sim.dashed(surface, sim.DASH, cue, G, sim.DASH_W)
                if PGCFG.SHOW_SOLID_LINES:
                    pygame.draw.line(surface, sim.LINE1, sim.px(cue), sim.px(G), 2)
            else:
                R = np.array(info.get("rail_pt"))
                if R is not None:
                    if PGCFG.SHOW_DASHED_GUIDES:
                        sim.dashed(surface, sim.DASH, cue, R, sim.DASH_W)
                        sim.dashed(surface, sim.DASH, R, G, sim.DASH_W)
                    if PGCFG.SHOW_SOLID_LINES:
                        pygame.draw.line(surface, sim.LINE1, sim.px(cue), sim.px(R), 2)
                        pygame.draw.line(surface, sim.LINE1, sim.px(R), sim.px(G), 2)
            # Target to pocket
            pocket_id = info.get("pocket_id")
            pkts_list = [np.array([0,0]), np.array([w/2,0]), np.array([w,0]),
                         np.array([0,h]), np.array([w/2,h]), np.array([w,h])]
            if pocket_id is not None and 0 <= pocket_id < len(pkts_list):
                PK = pkts_list[pocket_id]
                if PGCFG.SHOW_DASHED_GUIDES:
                    sim.dashed(surface, sim.DASH, target, PK, sim.DASH_W)
                if PGCFG.SHOW_SOLID_LINES:
                    pygame.draw.line(surface, sim.LINE2, sim.px(G), sim.px(target), 2)
                    pygame.draw.line(surface, sim.LINE2, sim.px(target), sim.px(PK), 2)

    return surface


def render_image(cue: Tuple[float, float], target: Tuple[float, float], blockers: List[Tuple[float, float]],
                 info: Optional[dict], out_dir: Path, labels: Optional[List[Tuple[str, Tuple[float, float]]]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    surf = _render_surface(cue, target, blockers, info, labels)
    # Save to PNG file
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"path_{ts}.png"
    pygame.image.save(surf, str(out_path))
    pygame.quit()
    return out_path

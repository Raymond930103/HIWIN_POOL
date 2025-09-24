import pygame, numpy as np
# Support package import (main.gui.simulator) and script import
try:
    from ..core.ball_generator import generate_layout
    from ..core.billiard_api import compute_shot
    from ..core.solver_core import BALL_R
except Exception:
    # Fallback for direct script execution
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core.ball_generator import generate_layout
    from core.billiard_api import compute_shot
    from core.solver_core import BALL_R
from main.configs.pygame_config import CONFIG as PGCFG
from main.configs.table import TABLE_M as TABLE

# --- 新增 ---
GRID_STEP  = 0.05            # m
GRID_COLOR = (110, 110, 110) # 深灰
GRID_WIDTH = 1               # px
LABEL_COLOR = (230, 230, 230)   # 淡灰（可自行換色）
LABEL_FONT  = None   # 18px 系統預設字



# ── 視覺參數 ───────────────────────────────────────────
# Keep SCALE/MARGIN local to visualization; TABLE from shared config (meters)
SCALE=800; MARGIN=20
R_BALL=int(BALL_R*SCALE); R_PK=int(R_BALL*1.6)
DASH_W=int(2*BALL_R*SCALE)

# Colors from config for backward-compat constants used elsewhere
GREEN = PGCFG.COLOR_TABLE
RAIL  = PGCFG.COLOR_RAIL
PKCOL = PGCFG.COLOR_POCKET
CUE   = PGCFG.COLOR_CUE_BALL
TARGET= PGCFG.COLOR_TARGET_BALL
OTH   = PGCFG.COLOR_OTHER_BALLS
LINE1 = PGCFG.COLOR_LINE_PRIMARY
LINE2 = PGCFG.COLOR_LINE_SECONDARY
DASH  = PGCFG.COLOR_DASHED

px=lambda p:(int(p[0]*SCALE+MARGIN),int(p[1]*SCALE+MARGIN))

def dashed(surf,col,a,b,w,dash=10,gap=6):
    ax,ay=px(a); bx,by=px(b); vec=np.array([bx-ax,by-ay],float)
    L=np.linalg.norm(vec); vec/=L
    n=int(L//(dash+gap))+1
    for i in range(n):
        st=np.array([ax,ay])+vec*i*(dash+gap); ed=st+vec*dash
        pygame.draw.line(surf,col,st,ed,int(w))


def draw_grid(surface, table_w, table_h):
    """在桌布上畫網格線並標註座標 (m)"""
    # 垂直線 +  x 數字
    x = 0.0
    while x <= table_w + 1e-6:
        px_x, _ = px((x, 0))
        pygame.draw.line(surface, GRID_COLOR,
                         (px_x, MARGIN),
                         (px_x, MARGIN + table_h * SCALE),
                         GRID_WIDTH)
        # 座標標籤（畫在桌布下方 5px）
        label = LABEL_FONT.render(f"{x:.2f}", True, LABEL_COLOR)
        rect  = label.get_rect(center=(px_x, MARGIN + table_h * SCALE + 12))
        surface.blit(label, rect)
        x += GRID_STEP

    # 水平線 +  y 數字
    y = 0.0
    while y <= table_h + 1e-6:
        _, px_y = px((0, y))
        pygame.draw.line(surface, GRID_COLOR,
                         (MARGIN, px_y),
                         (MARGIN + table_w * SCALE, px_y),
                         GRID_WIDTH)
        # 座標標籤（畫在桌布左側 5px）
        label = LABEL_FONT.render(f"{y:.2f}", True, LABEL_COLOR)
        rect  = label.get_rect(center=(MARGIN - 15, px_y))
        surface.blit(label, rect)
        y += GRID_STEP



# ── 主流程 ─────────────────────────────────────────────
def main():
    layout=generate_layout(n_blockers=3, seed=None)
    w,h   = layout['table']
    cue,tgt,blks=layout['cue'],layout['target'],layout['blockers']
    plan=compute_shot(cue,tgt,blks)
    print("plan =", plan) 

    pockets=[np.array([0,0]),np.array([w/2,0]),np.array([w,0]),
             np.array([0,h]),np.array([w/2,h]),np.array([w,h])]

    pygame.init()
    global LABEL_FONT
    LABEL_FONT = pygame.font.SysFont(None, 18)
    scr=pygame.display.set_mode((int(w*SCALE+MARGIN*2),
                                 int(h*SCALE+MARGIN*2)))
    pygame.display.set_caption("Billiard Path – demo")
    clock=pygame.time.Clock()

    run=True
    while run:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: run=False

        scr.fill(RAIL)
        pygame.draw.rect(scr,GREEN,(MARGIN,MARGIN,w*SCALE,h*SCALE))
        if PGCFG.SHOW_GRID:
            draw_grid(scr, w, h)
        if PGCFG.SHOW_POCKETS:
            for pk in pockets:
                pygame.draw.circle(scr,PKCOL,px(pk),R_PK)

        draw=lambda p,c:pygame.draw.circle(scr,c,px(p),R_BALL)
        # decide target color priority: force-normal > highlight > distinct
        if PGCFG.TARGET_AS_NORMAL:
            tgt_col = OTH
        elif PGCFG.HIGHLIGHT_TARGET_ON_PLAN and plan:
            tgt_col = PGCFG.COLOR_TARGET_HIGHLIGHT
        else:
            tgt_col = TARGET
        draw(cue,CUE); draw(tgt,tgt_col); [draw(b,OTH) for b in blks]

        if plan:
            G=np.array(plan['ghost'])
            PK=pockets[plan['pocket_id']]
            if plan['type']=='direct':
                if PGCFG.SHOW_DASHED_GUIDES:
                    dashed(scr,DASH,cue,G,DASH_W)
                if PGCFG.SHOW_SOLID_LINES:
                    pygame.draw.line(scr,LINE1,px(cue),px(G),2)
            else:
                R=np.array(plan['rail_pt'])
                if PGCFG.SHOW_DASHED_GUIDES:
                    dashed(scr,DASH,cue,R,DASH_W)
                    dashed(scr,DASH,R,G,DASH_W)
                if PGCFG.SHOW_SOLID_LINES:
                    pygame.draw.line(scr,LINE1,px(cue),px(R),2)
                    pygame.draw.line(scr,LINE1,px(R),px(G),2)

            if PGCFG.SHOW_DASHED_GUIDES:
                dashed(scr,DASH,tgt,PK,DASH_W)
            if PGCFG.SHOW_SOLID_LINES:
                pygame.draw.line(scr,LINE2,px(G),px(tgt),2)
                pygame.draw.line(scr,LINE2,px(tgt),px(PK),2)

        pygame.display.flip(); clock.tick(60)
    pygame.quit()

if __name__=='__main__':
    main()

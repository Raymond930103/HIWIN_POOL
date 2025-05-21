"""
顯示指定佈局 (cue, target, blockers) + compute_shot() 路徑
----------------------------------------------------------------
1. 直接 CLI：
       python -m gui.visualize --cue x y --target x y --blockers x1 y1 ...
2. 由其他程式呼叫：
       import gui.visualize as vis
       vis.show(cue, target, blockers, info)   # info 可省略
   這樣就不會因為 argparse 卡住。
"""
import argparse, numpy as np, pygame
from core.billiard_api import compute_shot
import gui.simulator as sim   # 所有視覺常數 / 函式

# ──────────────────────────────────────────────────────────────
# Internal: 單純把顯示流程包成函式，讓 CLI 與 show() 共用
# ──────────────────────────────────────────────────────────────

def _render(cue, target, blockers, info):
    """在 Pygame 視窗即時顯示路徑，直到關閉窗口"""

    pygame.init()
    # 給 simulator 用的字型（draw_grid 會用到）
    sim.LABEL_FONT = pygame.font.SysFont(None, 18)

    scr = pygame.display.set_mode((
        int(sim.TABLE[0]*sim.SCALE + sim.MARGIN*2),
        int(sim.TABLE[1]*sim.SCALE + sim.MARGIN*2)
    ))
    pygame.display.set_caption("compute_shot visualize")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 48)

    # 袋口世界座標
    w, h = sim.TABLE
    pkts = [np.array([0,0]), np.array([w/2,0]), np.array([w,0]),
            np.array([0,h]), np.array([w/2,h]), np.array([w,h])]

    # ---- 主迴圈 --------------------------------------------------
    run = True
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

        # --- 背景 & 桌布 ---
        scr.fill(sim.RAIL)
        pygame.draw.rect(scr, sim.GREEN,
                         (sim.MARGIN, sim.MARGIN,
                          w*sim.SCALE, h*sim.SCALE))
        # --- 網格線 + 座標 ---
        sim.draw_grid(scr, w, h)

        # --- 袋口 ---
        for pk in pkts:
            pygame.draw.circle(scr, sim.PKCOL, sim.px(pk), sim.R_PK)

        # --- 球 ---
        draw_ball = lambda p,c: pygame.draw.circle(scr, c, sim.px(p), sim.R_BALL)
        draw_ball(cue,    sim.CUE)
        draw_ball(target, sim.TARGET)
        for b in blockers:
            draw_ball(b, sim.OTH)

        # --- 路徑 / NO PATH ---
        if info:
            G  = np.array(info["ghost"])
            PK = pkts[info["pocket_id"]]
            if info["type"] == "direct":
                sim.dashed(scr, sim.DASH, cue, G, sim.DASH_W)
                pygame.draw.line(scr, sim.LINE1, sim.px(cue), sim.px(G), 2)
            else:
                R = np.array(info["rail_pt"])
                sim.dashed(scr, sim.DASH, cue, R, sim.DASH_W)
                sim.dashed(scr, sim.DASH, R, G, sim.DASH_W)
                pygame.draw.line(scr, sim.LINE1, sim.px(cue), sim.px(R), 2)
                pygame.draw.line(scr, sim.LINE1, sim.px(R), sim.px(G), 2)

            sim.dashed(scr, sim.DASH, target, PK, sim.DASH_W)
            pygame.draw.line(scr, sim.LINE2, sim.px(G), sim.px(target), 2)
            pygame.draw.line(scr, sim.LINE2, sim.px(target), sim.px(PK), 2)
        else:
            txt = font.render("NO  PATH", True, (255,0,0))
            rect = txt.get_rect(center=(scr.get_width()/2, scr.get_height()/2))
            scr.blit(txt, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ──────────────────────────────────────────────────────────────
# Public: 供外部程式呼叫
# ──────────────────────────────────────────────────────────────

def show(cue, target, blockers, info=None):
    """外部 API：丟座標進來就跑視覺化
    cue/target: (x,y) in m
    blockers  : [(x,y), ...] in m
    info      : compute_shot() 回傳 dict，可省略 → 內部自算
    """
    if info is None:
        info = compute_shot(cue, target, blockers)
    _render(np.asarray(cue), np.asarray(target),
            [np.asarray(b) for b in blockers], info)


# ──────────────────────────────────────────────────────────────
# CLI 入口：只有在直接執行時才跑 argparse
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("visualize compute_shot result")
    ap.add_argument("--cue",    nargs=2, type=float, required=True)
    ap.add_argument("--target", nargs=2, type=float, required=True)
    ap.add_argument("--blockers", nargs="*", type=float, default=[])
    args = ap.parse_args()

    cue      = tuple(args.cue)
    target   = tuple(args.target)
    blockers = list(zip(args.blockers[::2], args.blockers[1::2]))

    info = compute_shot(cue, target, blockers)
    print("compute_shot ⇒", info)

    _render(np.asarray(cue), np.asarray(target),
            [np.asarray(b) for b in blockers], info)

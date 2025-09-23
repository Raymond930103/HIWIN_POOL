import pygame, numpy as np
import argparse
from core.ball_generator import generate_layout
from core.billiard_api  import compute_shot
from core.solver_core   import BALL_R
from main.configs.pygame_config import CONFIG as PGCFG

# ── 視覺參數 ───────────────────────────────────────────
TABLE = (0.73, 0.375); SCALE=400; MARGIN=20
R_BALL=int(BALL_R*SCALE); R_PK=int(R_BALL*1.6)
DASH_W=int(2*BALL_R*SCALE)

# Colors from config
GREEN=PGCFG.COLOR_TABLE; RAIL=PGCFG.COLOR_RAIL; PKCOL=PGCFG.COLOR_POCKET
CUE=PGCFG.COLOR_CUE_BALL; TARGET=PGCFG.COLOR_TARGET_BALL; OTH=PGCFG.COLOR_OTHER_BALLS
LINE1=PGCFG.COLOR_LINE_PRIMARY; LINE2=PGCFG.COLOR_LINE_SECONDARY; DASH=PGCFG.COLOR_DASHED
px=lambda p:(int(p[0]*SCALE+MARGIN),int(p[1]*SCALE+MARGIN))

def dashed(surf,col,a,b,w,dash=10,gap=6):
    ax,ay=px(a); bx,by=px(b); vec=np.array([bx-ax,by-ay],float)
    L=np.linalg.norm(vec); vec/=L
    n=int(L//(dash+gap))+1
    for i in range(n):
        st=np.array([ax,ay])+vec*i*(dash+gap); ed=st+vec*dash
        pygame.draw.line(surf,col,st,ed,int(w))

# ── 主流程 ─────────────────────────────────────────────
def main():
    # optional CLI override
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--target-as-normal", action="store_true")
    ap.add_argument("--no-highlight", action="store_true")
    args, _ = ap.parse_known_args()
    if args.target_as_normal:
        PGCFG.TARGET_AS_NORMAL = True
    if args.no_highlight:
        PGCFG.HIGHLIGHT_TARGET_ON_PLAN = False

    layout=generate_layout(n_blockers=3, seed=None)
    w,h   = layout['table']
    cue,tgt,blks=layout['cue'],layout['target'],layout['blockers']
    plan=compute_shot(cue,tgt,blks)

    pockets=[np.array([0,0]),np.array([w/2,0]),np.array([w,0]),
             np.array([0,h]),np.array([w/2,h]),np.array([w,h])]

    pygame.init()
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
        if PGCFG.SHOW_POCKETS:
            for pk in pockets: pygame.draw.circle(scr,PKCOL,px(pk),R_PK)

        draw=lambda p,c:pygame.draw.circle(scr,c,px(p),R_BALL)
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

import pygame, numpy as np
from core.ball_generator import generate_layout
from core.billiard_api  import compute_shot
from core.solver_core   import BALL_R

# ── 視覺參數 ───────────────────────────────────────────
TABLE = (0.73, 0.375); SCALE=400; MARGIN=20
R_BALL=int(BALL_R*SCALE); R_PK=int(R_BALL*1.6)
DASH_W=int(2*BALL_R*SCALE)

GREEN=(18,95,29); RAIL=(60,30,10); PKCOL=(25,12,4)
CUE=(245,245,245); TARGET=(255,90,40); OTH=(40,120,255)
LINE1=(250,0,0); LINE2=(255,255,0); DASH=(185,185,185)
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
        for pk in pockets: pygame.draw.circle(scr,PKCOL,px(pk),R_PK)

        draw=lambda p,c:pygame.draw.circle(scr,c,px(p),R_BALL)
        draw(cue,CUE); draw(tgt,TARGET); [draw(b,OTH) for b in blks]

        if plan:
            G=np.array(plan['ghost'])
            PK=pockets[plan['pocket_id']]
            if plan['type']=='direct':
                dashed(scr,DASH,cue,G,DASH_W)
                pygame.draw.line(scr,LINE1,px(cue),px(G),2)
            else:
                R=np.array(plan['rail_pt'])
                dashed(scr,DASH,cue,R,DASH_W)
                dashed(scr,DASH,R,G,DASH_W)
                pygame.draw.line(scr,LINE1,px(cue),px(R),2)
                pygame.draw.line(scr,LINE1,px(R),px(G),2)

            dashed(scr,DASH,tgt,PK,DASH_W)
            pygame.draw.line(scr,LINE2,px(G),px(tgt),2)
            pygame.draw.line(scr,LINE2,px(tgt),px(PK),2)

        pygame.display.flip(); clock.tick(60)
    pygame.quit()

if __name__=='__main__':
    main()

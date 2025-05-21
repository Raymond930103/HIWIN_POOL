import math, numpy as np

BALL_R = 0.0125          # (m) 花式撞球半徑
EPS    = 1e-9
norm   = np.linalg.norm
dist   = lambda a, b: norm(a - b)

def angle(u, v):
    nu, nv = norm(u), norm(v)
    if nu < EPS or nv < EPS:
        return math.pi
    c = np.dot(u, v) / (nu * nv)
    return math.acos(max(-1.0, min(1.0, c)))

def path_clear(p1, p2, balls, *, ignore=frozenset(),
               rail=False, table=None):
    """線段 p1→p2 在『中心 ±2R』走廊內是否碰球 / 撞牆"""
    v, L = p2 - p1, norm(p2 - p1)
    if L < EPS: return False
    d = v / L

    # 球
    for b in balls:
        if b['id'] in ignore: continue
        proj = np.clip(np.dot(b['pos'] - p1, d), 0.0, L)
        close = p1 + proj * d
        if dist(close, b['pos']) < 2*BALL_R - 1e-4:
            return False

    # Rail
    if rail and table is not None:
        W, H = table
        minx, maxx = sorted((p1[0], p2[0]))
        miny, maxy = sorted((p1[1], p2[1]))
        if (minx < BALL_R - 1e-4 or maxx > W-BALL_R+1e-4 or
            miny < BALL_R - 1e-4 or maxy > H-BALL_R+1e-4):
            return False
    return True


class BilliardSolver:
    """幾何求解：直球優先，若被擋→單庫反彈"""
    def __init__(self, table_size, pockets):
        self.W, self.H = table_size
        self.pockets   = pockets

    # ── API ───────────────────────────────────────
    def solve(self, cue, tgt, others):
        balls = [{'id':0,'pos':cue}, {'id':1,'pos':tgt}] + \
                [{'id':i+2,'pos':p} for i,p in enumerate(others)]

        v_ct = tgt - cue
        order = sorted(self.pockets,
                       key=lambda pk: (angle(v_ct, pk-tgt),
                                       dist(pk, tgt)))

        for pk in order:
            G = self._ghost(tgt, pk)
            if not self._inside(G): continue

            ok_CG = path_clear(cue, G, balls,
                               ignore={0,1},
                               rail=True, table=(self.W, self.H))
            ok_TP = path_clear(tgt, pk, balls, ignore={1})

            if ok_CG and ok_TP:
                return {'type':'direct', 'pocket':pk, 'ghost':G}

            for R in self._mirror(G):
                ok_CR = path_clear(cue, R, balls, ignore={0,1})
                ok_RG = path_clear(R,  G, balls, ignore={0,1})
                if ok_CR and ok_RG and ok_TP:
                    return {'type':'bank-1',
                            'pocket':pk, 'ghost':G, 'rail_pt':R}
        return None

    # ── 私有 ───────────────────────────────────────
    def _ghost(self, T, P):
        v = T - P; v /= norm(v)
        return T + v * 2*BALL_R

    def _inside(self, p):
        return BALL_R <= p[0] <= self.W-BALL_R and \
               BALL_R <= p[1] <= self.H-BALL_R

    def _mirror(self, G):
        x,y = G; W,H = self.W, self.H
        return [np.array([-x,   y]), np.array([2*W-x, y]),
                np.array([x,  -y]), np.array([x, 2*H-y])]

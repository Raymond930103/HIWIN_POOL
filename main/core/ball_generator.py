import random, numpy as np
from .solver_core import BALL_R

def _rand(w, h):
    return np.array([random.uniform(BALL_R, w-BALL_R),
                     random.uniform(BALL_R, h-BALL_R)])

def generate_layout(table_size=(0.73, 0.375),
                    n_blockers=2, seed=None):
    if seed is not None: random.seed(seed)
    w,h = table_size
    cue = _rand(w,h)
    while True:
        tgt = _rand(w,h)
        if np.linalg.norm(tgt-cue) > 4*BALL_R: break

    blockers=[]
    while len(blockers)<n_blockers:
        p=_rand(w,h)
        if (np.linalg.norm(p-cue)>4*BALL_R and
            np.linalg.norm(p-tgt)>4*BALL_R and
            all(np.linalg.norm(p-q)>4*BALL_R for q in blockers)):
            blockers.append(p)
    return {'table':table_size,'cue':cue,
            'target':tgt,'blockers':blockers}

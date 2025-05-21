import math, numpy as np
from .solver_core import BilliardSolver

TABLE = (0.735, 0.375)  # (m) 桌面尺寸
POCKETS = [np.array([0,0]),
           np.array([TABLE[0]/2, 0]),
           np.array([TABLE[0],   0]),
           np.array([0,  TABLE[1]]),
           np.array([TABLE[0]/2, TABLE[1]]),
           np.array([TABLE[0],   TABLE[1]])]

_solver = BilliardSolver(TABLE, POCKETS)

def compute_shot(cue, target, blockers):
    cue     = np.asarray(cue,    dtype=float)
    target  = np.asarray(target, dtype=float)
    blockers= [np.asarray(b, dtype=float) for b in blockers]

    plan = _solver.solve(cue, target, blockers)
    if plan is None:
        return None

    
    if plan['type'] == 'direct':          # 直球：對準 ghost ball
        v = plan['ghost'] - cue
    elif plan['type'] == 'bank-1':        # 單庫：先打到 rail_pt
        v = plan['rail_pt'] - cue
    else:                                 # 其他類型保留原邏輯
        v = plan['ghost'] - cue

    angle_deg = round(math.degrees(math.atan2(v[1], v[0])), 2)
    
    
    pk_id = next(i for i, pk in enumerate(POCKETS)
                if np.allclose(pk, plan['pocket']))

    res = {'type':plan['type'], 'pocket_id':pk_id,
           'ghost':[round(x,4) for x in plan['ghost']],
           'angle_deg':angle_deg}
    if plan['type']=='bank-1':
        res['rail_pt']=[round(x,4) for x in plan['rail_pt']]
    return res

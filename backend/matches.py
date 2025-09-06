from datetime import datetime
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from . import models, schemas, database
from .auth import get_current_user
from main.core.billiard_api import compute_shot
from main.communicate import tcp_communicate

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=schemas.MatchOut)
def create_match(match_in: schemas.MatchCreate,
                 db: Session = Depends(database.get_db),
                 user: models.User = Depends(get_current_user)):
    match = models.Match(user_id=user.id,
                         mode=match_in.mode,
                         difficulty=match_in.difficulty,
                         start_time=datetime.utcnow())
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@router.post("/{match_id}/rounds", response_model=schemas.RoundOut)
def create_round(match_id: int,
                 round_in: schemas.RoundCreate,
                 db: Session = Depends(database.get_db),
                 user: models.User = Depends(get_current_user)):
    match = db.query(models.Match).filter(models.Match.id == match_id,
                                          models.Match.user_id == user.id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    rnd = models.Round(match_id=match.id,
                       turn=round_in.turn,
                       shot_result=round_in.shot_result,
                       state_json=round_in.state_json)
    db.add(rnd)
    db.commit()
    db.refresh(rnd)

    if round_in.turn == models.Turn.player:
        try:
            state = json.loads(round_in.state_json)
            info = compute_shot(state['cue'], state['target'], state.get('others', []))
            if info:
                tcp_communicate.send_shot(info['angle_deg'], 0.8, 0.0)
                ai_rnd = models.Round(match_id=match.id,
                                      turn=models.Turn.ai,
                                      shot_result=models.ShotResult.miss,
                                      state_json=round_in.state_json)
                db.add(ai_rnd)
                db.commit()
        except Exception:
            pass
    return rnd


@router.get("/{match_id}", response_model=schemas.MatchOut)
def get_match(match_id: int,
              db: Session = Depends(database.get_db),
              user: models.User = Depends(get_current_user)):
    match = db.query(models.Match).filter(models.Match.id == match_id,
                                          models.Match.user_id == user.id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match

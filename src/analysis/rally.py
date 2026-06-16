"""Segmentación de rallies y atribución de errores."""
from __future__ import annotations

from typing import List, Optional

from ..datatypes import Bounce, Shot, Rally
from ..utils.logging_utils import get_logger
from .court import CourtModel

logger = get_logger(__name__)


class RallySegmenter:
    """
    Agrupa los golpes en rallies y decide cómo termina cada uno:

      - "double_bounce": dos botes seguidos en el mismo lado sin golpe entre
        medias -> el jugador de ese lado no llegó a la pelota (pierde el punto).
      - "out": el último bote del rally cae fuera de la pista -> el último
        jugador que golpeó cometió el error.
      - "gap": la pelota deja de detectarse demasiados frames.

    El error se atribuye al jugador que pierde el punto.
    """

    def __init__(self, court: CourtModel, max_gap_frames: int = 25):
        self.court = court
        self.max_gap = max_gap_frames

    def segment(self, shots: List[Shot], bounces: List[Bounce],
                double_bounce_frames: List[int] = None) -> List[Rally]:
        if not shots:
            return []
        db = sorted(double_bounce_frames or [])

        rallies: List[Rally] = []
        current: List[Shot] = [shots[0]]

        for prev, nxt in zip(shots, shots[1:]):
            gap = nxt.frame - prev.frame
            # Fin de rally: hueco largo entre golpes O un doble bote (juego parado)
            stopped = any(prev.frame < f <= nxt.frame for f in db)
            if gap > self.max_gap or stopped:
                rallies.append(self._close(current, bounces))
                current = [nxt]
            else:
                current.append(nxt)
        rallies.append(self._close(current, bounces))
        return rallies

    # ------------------------------------------------------------------
    def _close(self, shots: List[Shot], bounces: List[Bounce]) -> Rally:
        start, end = shots[0].frame, shots[-1].frame
        rally = Rally(start_frame=start, end_frame=end, shots=list(shots))

        # Botes posteriores al último golpe -> determinan el final
        post = [b for b in bounces if b.frame >= end]
        last_shooter = shots[-1].player_id

        ended_by = "gap"
        error_player: Optional[int] = last_shooter

        if post:
            first_after = post[0]
            if not first_after.inside:
                ended_by = "out"
                error_player = last_shooter        # golpeó fuera
            else:
                # ¿Doble bote en el mismo lado? -> el rival no devolvió
                same_side = [b for b in post if b.side == first_after.side]
                if len(same_side) >= 2:
                    ended_by = "double_bounce"
                    # error del jugador del lado donde botó dos veces (no devolvió)
                    error_player = self._player_on_side(shots, first_after.side)

        rally.ended_by = ended_by
        rally.error_player = error_player
        logger.debug("Rally frames %d-%d | %d golpes | fin=%s | error=jugador %s",
                     start, end, len(shots), ended_by, error_player)
        return rally

    def _player_on_side(self, shots: List[Shot], side: str) -> Optional[int]:
        """Jugador asociado al lado de pista indicado (por sus impactos)."""
        for s in reversed(shots):
            if self.court.side_of(s.court_y) == side:
                return s.player_id
        return shots[-1].player_id if shots else None

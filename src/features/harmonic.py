"""
Harmonic mixing logic for the Transition Playlist feature.

Uses the Camelot Wheel — the industry standard for DJ harmonic mixing.
Two tracks are harmonically compatible if their Camelot codes are:
  - The same code (same key/mode)
  - Adjacent numbers (±1) on the wheel
  - The same number, opposite letter (A ↔ B = relative major/minor)

Pitch Class → Camelot mapping:
  Key (0=C, 1=C#, ..., 11=B) × Mode (0=minor, 1=major)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (key, mode) → Camelot code  e.g. (0, 1) = "8B" (C major)
PITCH_TO_CAMELOT: dict[tuple[int, int], str] = {
    (0,  1): "8B",   # C major
    (1,  1): "3B",   # C# major
    (2,  1): "10B",  # D major
    (3,  1): "5B",   # Eb major
    (4,  1): "12B",  # E major
    (5,  1): "7B",   # F major
    (6,  1): "2B",   # F# major
    (7,  1): "9B",   # G major
    (8,  1): "4B",   # Ab major
    (9,  1): "11B",  # A major
    (10, 1): "6B",   # Bb major
    (11, 1): "1B",   # B major
    (0,  0): "5A",   # C minor
    (1,  0): "12A",  # C# minor
    (2,  0): "7A",   # D minor
    (3,  0): "2A",   # Eb minor
    (4,  0): "9A",   # E minor
    (5,  0): "4A",   # F minor
    (6,  0): "11A",  # F# minor
    (7,  0): "6A",   # G minor
    (8,  0): "1A",   # Ab minor
    (9,  0): "8A",   # A minor
    (10, 0): "3A",   # Bb minor
    (11, 0): "10A",  # B minor
}


def get_camelot(key: int, mode: int) -> str | None:
    return PITCH_TO_CAMELOT.get((int(key), int(mode)))


def camelot_compatible(code_a: str, code_b: str) -> bool:
    """True if two Camelot codes are harmonically mixable."""
    if code_a is None or code_b is None:
        return False
    num_a, letter_a = int(code_a[:-1]), code_a[-1]
    num_b, letter_b = int(code_b[:-1]), code_b[-1]

    same_letter   = letter_a == letter_b
    adj_number    = abs(num_a - num_b) <= 1 or abs(num_a - num_b) == 11  # wheel wraps at 12
    same_number   = num_a == num_b

    return (same_letter and adj_number) or (same_number)  # same number diff letter = relative


def bpm_compatible(tempo_a: float, tempo_b: float, tolerance: float = 5.0) -> bool:
    """True if BPMs are within tolerance (or one is double/half the other)."""
    if abs(tempo_a - tempo_b) <= tolerance:
        return True
    # Harmonic doubling (80 BPM → 160 BPM is perfectly mixable)
    if abs(tempo_a - tempo_b * 2) <= tolerance or abs(tempo_a * 2 - tempo_b) <= tolerance:
        return True
    return False


def transition_score(
    track_a: pd.Series,
    track_b: pd.Series,
    bpm_tolerance: float = 5.0,
) -> float:
    """
    Composite transition score between two tracks (0.0 – 1.0).
    Higher = smoother transition.

    Components:
      - Harmonic compatibility (40%)
      - BPM compatibility (30%)
      - Energy delta (20%)
      - Valence delta (10%)
    """
    score = 0.0

    # Harmonic (40%)
    camelot_a = get_camelot(track_a.get("key", -1), track_a.get("mode", -1))
    camelot_b = get_camelot(track_b.get("key", -1), track_b.get("mode", -1))
    if camelot_compatible(camelot_a, camelot_b):
        score += 0.40

    # BPM (30%)
    tempo_a = float(track_a.get("tempo", 0))
    tempo_b = float(track_b.get("tempo", 0))
    if tempo_a > 0 and tempo_b > 0:
        if bpm_compatible(tempo_a, tempo_b, bpm_tolerance):
            bpm_delta = min(abs(tempo_a - tempo_b), bpm_tolerance)
            score += 0.30 * (1 - bpm_delta / bpm_tolerance)

    # Energy delta (20%) — closer energy = smoother transition
    energy_a = float(track_a.get("energy", 0.5))
    energy_b = float(track_b.get("energy", 0.5))
    score += 0.20 * (1 - abs(energy_a - energy_b))

    # Valence delta (10%)
    valence_a = float(track_a.get("valence", 0.5))
    valence_b = float(track_b.get("valence", 0.5))
    score += 0.10 * (1 - abs(valence_a - valence_b))

    return round(score, 4)


def build_transition_graph(
    tracks: pd.DataFrame,
    bpm_tolerance: float = 5.0,
) -> dict[int, list[tuple[int, float]]]:
    """
    Build adjacency list where edges connect harmonically/BPM-compatible tracks.
    Edge weight = transition_score.
    Only considers pairs with score > 0.5 to keep graph sparse.

    Returns: {track_idx: [(neighbor_idx, score), ...]}
    """
    graph: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(tracks))}

    tempos   = tracks["tempo"].values   if "tempo"  in tracks.columns else np.zeros(len(tracks))
    keys     = tracks["key"].values     if "key"    in tracks.columns else np.zeros(len(tracks))
    modes    = tracks["mode"].values    if "mode"   in tracks.columns else np.zeros(len(tracks))
    energies = tracks["energy"].values  if "energy" in tracks.columns else np.full(len(tracks), 0.5)
    valences = tracks["valence"].values if "valence" in tracks.columns else np.full(len(tracks), 0.5)

    # Precompute Camelot codes
    camelots = [get_camelot(int(k), int(m)) for k, m in zip(keys, modes)]

    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i == j:
                continue
            # Fast pre-filter: BPM must be at least close
            if not bpm_compatible(float(tempos[i]), float(tempos[j]), bpm_tolerance * 2):
                continue

            a = {"key": keys[i], "mode": modes[i], "tempo": tempos[i],
                 "energy": energies[i], "valence": valences[i]}
            b = {"key": keys[j], "mode": modes[j], "tempo": tempos[j],
                 "energy": energies[j], "valence": valences[j]}

            score = transition_score(a, b, bpm_tolerance)
            if score >= 0.5:
                graph[i].append((j, score))

    return graph

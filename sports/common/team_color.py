"""
Team split by shirt colour.

`TeamClassifier` (SigLIP embeddings) works well on broadcast footage, where
players are 60+ pixels tall. From a wide camera on a training pitch they are
often 15-30 pixels tall and the embeddings mix the two teams. The shirt colour
is still clear at that size, so this classifier clusters the median colour of
the shirt area, after removing the grass around the player.
"""
from typing import List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


def shirt_color(crop: np.ndarray) -> Optional[np.ndarray]:
    """Median Lab colour of the shirt area of a BGR player crop, or None."""
    if crop is None or crop.ndim != 3:
        return None
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # the crop border is mostly grass: its colour tells what to remove
    border = np.concatenate([lab[:, 0], lab[:, -1], lab[0], lab[-1]])
    grass = np.median(border, axis=0)
    torso = lab[int(.15 * h):max(int(.5 * h), int(.15 * h) + 1),
                int(.2 * w):max(int(.8 * w), int(.2 * w) + 1)].reshape(-1, 3)
    keep = torso[np.linalg.norm(torso[:, 1:] - grass[1:], axis=1) > 10]
    if len(keep) < 5:
        keep = torso
    return np.median(keep, axis=0)


class ColorTeamClassifier:
    """
    Same `fit` / `predict` interface as `TeamClassifier`.

    Example:
        ```python
        classifier = ColorTeamClassifier().fit(crops)
        teams = classifier.predict(crops)      # 0 or 1
        far = classifier.distance(crops) > classifier.threshold  # neither team
        ```
    """

    def __init__(self, n_teams: int = 2, brightness_weight: float = 0.5) -> None:
        self.n_teams = n_teams
        self.brightness_weight = brightness_weight
        self.cluster_model: Optional[KMeans] = None
        self.threshold = np.inf

    def features(self, crops: List[np.ndarray]) -> np.ndarray:
        rows = []
        for crop in crops:
            c = shirt_color(crop)
            rows.append(np.full(3, np.nan) if c is None else c)
        f = np.array(rows, dtype=np.float32).reshape(-1, 3)
        f[:, 0] *= self.brightness_weight  # shade and distance change brightness more than hue
        return f

    def fit(self, crops: List[np.ndarray]) -> "ColorTeamClassifier":
        f = self.features(crops)
        f = f[~np.isnan(f).any(axis=1)]
        if len(f) < self.n_teams:
            raise ValueError("not enough player crops to learn the team colours")
        self.cluster_model = KMeans(self.n_teams, n_init=10, random_state=0).fit(f)
        d = self._distances(f).min(axis=1)
        # beyond this distance a shirt belongs to neither team (referee, staff)
        self.threshold = float(2 * np.percentile(d, 90))
        return self

    def _distances(self, f: np.ndarray) -> np.ndarray:
        return np.linalg.norm(f[:, None] - self.cluster_model.cluster_centers_[None], axis=2)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([], dtype=int)
        f = np.nan_to_num(self.features(crops), nan=0.0)
        return self._distances(f).argmin(axis=1).astype(int)

    def distance(self, crops: List[np.ndarray]) -> np.ndarray:
        """Distance from each crop's shirt colour to the nearest team colour."""
        if len(crops) == 0:
            return np.array([], dtype=float)
        f = self.features(crops)
        d = np.full(len(f), np.inf)
        ok = ~np.isnan(f).any(axis=1)
        if ok.any():
            d[ok] = self._distances(f[ok]).min(axis=1)
        return d

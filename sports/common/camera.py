"""
Pitch calibration for a fixed camera that pans and zooms.

The pitch keypoint model in this repository was trained on broadcast footage.
On a low camera beside a training pitch it finds the wrong points, so players
are projected to the wrong place. When the camera always stands in the same
spot, a better option is to calibrate it once, by hand, on one reference
image, and then follow the camera movement frame by frame:

    frame --(ORB feature matching, RANSAC)--> reference image --(fixed H)--> pitch

A camera that only rotates and zooms around its own centre (a panning camera
on a tripod) maps between any two of its images with a homography, whatever
the depth of the scene, so the trees, buildings and stands behind the pitch
are as good as the pitch lines for this.
"""
import json
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class HomographyTransformer:
    """Same interface as `ViewTransformer`: image points -> pitch points."""

    def __init__(self, m: npt.NDArray[np.float64]) -> None:
        self.m = np.asarray(m, dtype=np.float64)

    def transform_points(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(-1, 2)
        out = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), self.m)
        return out.reshape(-1, 2).astype(np.float32)


def _mask(shape: Tuple[int, int], ignore: Sequence[Sequence[int]]) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in ignore:
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    return mask


class CameraTracker:
    """
    Follows a panning/zooming camera against a calibrated reference image.

    Example:
        ```python
        tracker = CameraTracker.from_file("calibracao.json")
        for frame in frames:
            transformer = tracker.update(frame)
            if transformer is not None:
                pitch_xy = transformer.transform_points(xy)
        ```
    """

    def __init__(
        self,
        reference: np.ndarray,
        image_to_pitch: npt.NDArray[np.float64],
        ignore: Sequence[Sequence[int]] = (),
        n_features: int = 3000,
        min_inliers_reference: int = 60,
        min_inliers_keyframe: int = 30,
        max_age: int = 30
    ) -> None:
        """
        Args:
            reference: BGR reference image the calibration was made on.
            image_to_pitch: 3x3 homography, reference pixels -> pitch (cm).
            ignore: Rectangles (x0, y0, x1, y1) with overlays that do not move
                with the camera (scoreboards, player UI), left out of matching.
            min_inliers_reference: Matches needed to trust a direct match to
                the reference image.
            min_inliers_keyframe: Matches needed to chain through the last
                good frame when the reference is out of view.
            max_age: Failed frames for which the last good result is reused.
        """
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ignore = [tuple(map(int, r)) for r in ignore]
        self.image_to_pitch = np.asarray(image_to_pitch, dtype=np.float64)
        self.min_inliers_reference = min_inliers_reference
        self.min_inliers_keyframe = min_inliers_keyframe
        self.max_age = max_age
        self.reference = self._features(reference)
        self.reset()

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "CameraTracker":
        """Loads a calibration JSON (see `save_calibration`)."""
        with open(path, encoding="utf-8") as fh:
            cal = json.load(fh)
        image = cv2.imread(os.path.join(os.path.dirname(path), cal["imagem"]))
        if image is None:
            raise FileNotFoundError(cal["imagem"])
        kwargs.setdefault("ignore", cal.get("ignorar_px", ()))
        tracker = cls(image, cal["homografia_px_para_cm"], **kwargs)
        tracker.pitch_length_m = cal.get("comprimento_m")
        tracker.pitch_width_m = cal.get("largura_m")
        return tracker

    def reset(self) -> None:
        self._key = None          # (features, frame -> reference homography)
        self._last = None
        self._age = 10 ** 9

    def _features(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return self.orb.detectAndCompute(gray, _mask(gray.shape, self.ignore))

    def _match(self, a, b) -> Tuple[Optional[np.ndarray], int]:
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, 0
        pairs = self.matcher.knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return None, 0
        pa = np.float32([ka[m.queryIdx].pt for m in good])
        pb = np.float32([kb[m.trainIdx].pt for m in good])
        h, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
        if h is None or inliers is None:
            return None, 0
        return h, int(inliers.sum())

    def frame_to_reference(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Homography from this frame to the reference image, or None."""
        cur = self._features(frame)
        h, n = self._match(cur, self.reference)
        if h is not None and n >= self.min_inliers_reference:
            self._key = (cur, h)
            return h
        if self._key is not None:
            h, n = self._match(cur, self._key[0])
            if h is not None and n >= self.min_inliers_keyframe:
                h = self._key[1] @ h
                self._key = (cur, h)
                return h
        return None

    def compute(self, frame: np.ndarray) -> Optional[HomographyTransformer]:
        h = self.frame_to_reference(frame)
        return None if h is None else HomographyTransformer(self.image_to_pitch @ h)

    def update(self, frame: np.ndarray) -> Optional[HomographyTransformer]:
        """Calibration for this frame, falling back to a recent good one."""
        transformer = self.compute(frame)
        if transformer is not None:
            self._last, self._age = transformer, 0
            return transformer
        self._age += 1
        return self._last if self._age <= self.max_age else None


def save_calibration(
    path: str,
    reference: np.ndarray,
    image_to_pitch: npt.NDArray[np.float64],
    length_m: float,
    width_m: float,
    ignore: Sequence[Sequence[int]] = (),
    description: str = ""
) -> None:
    """Writes `path` (JSON) and the reference image next to it."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
    cv2.imwrite(os.path.join(folder, name), reference, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"descricao": description, "imagem": name,
                   "comprimento_m": round(float(length_m), 1),
                   "largura_m": round(float(width_m), 1),
                   "homografia_px_para_cm": np.asarray(image_to_pitch).tolist(),
                   "ignorar_px": [list(map(int, r)) for r in ignore]},
                  fh, indent=1, ensure_ascii=False)

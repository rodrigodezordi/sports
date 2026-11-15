from typing import Optional

import numpy as np
import numpy.typing as npt

from sports.configs.soccer import SoccerPitchConfiguration


def points_well_spread(
    pitch_points: npt.NDArray[np.float32],
    min_std: float = 300.0
) -> bool:
    """
    Check that pitch reference points are spread in both directions.

    A homography computed from points that all lie on one line (for example,
    only the halfway line is visible) is badly conditioned and projects
    players to the wrong place, even when there are four or more points.

    Args:
        pitch_points (npt.NDArray[np.float32]): Points in pitch coordinates,
            shape (N, 2).
        min_std (float): Minimum standard deviation, in pitch units (cm for
            `SoccerPitchConfiguration`), along the narrowest direction of the
            point cloud. Defaults to 300 (3 m).

    Returns:
        bool: True when there are at least 4 points and they span an area.
    """
    pitch_points = np.asarray(pitch_points, dtype=float)
    if pitch_points.ndim != 2 or len(pitch_points) < 4:
        return False
    centered = pitch_points - pitch_points.mean(axis=0)
    smallest = np.linalg.svd(centered, compute_uv=False)[-1]
    return smallest / np.sqrt(len(pitch_points)) >= min_std


class PitchCalibrator:
    """
    Turns pitch keypoint detections into an image -> pitch `ViewTransformer`,
    rejecting unreliable frames and reusing the last good calibration for a
    short time (useful when the camera briefly shows too few pitch lines).

    Example:
        ```python
        calibrator = PitchCalibrator(max_age=int(fps))
        for frame in frames:
            key_points = sv.KeyPoints.from_inference(model.infer(frame)[0])
            transformer = calibrator.update(key_points)
            if transformer is not None:
                pitch_xy = transformer.transform_points(xy)
        ```
    """

    def __init__(
        self,
        config: Optional[SoccerPitchConfiguration] = None,
        confidence_threshold: float = 0.5,
        min_std: float = 300.0,
        max_age: int = 30
    ) -> None:
        """
        Args:
            config (Optional[SoccerPitchConfiguration]): Pitch geometry whose
                vertices match the keypoint model. Defaults to the standard one.
            confidence_threshold (float): Minimum keypoint confidence.
            min_std (float): See `points_well_spread`.
            max_age (int): How many consecutive failed updates the last good
                calibration is still returned for. 0 disables reuse.
        """
        self.config = config or SoccerPitchConfiguration()
        self.confidence_threshold = confidence_threshold
        self.min_std = min_std
        self.max_age = max_age
        self.reset()

    def reset(self) -> None:
        self._last = None
        self._age = 10 ** 9

    def compute(self, key_points) -> Optional["ViewTransformer"]:
        """Calibration from this frame only, or None if it is not reliable."""
        from sports.common.view import ViewTransformer

        if key_points is None or len(key_points.xy) == 0:
            return None
        confidence = getattr(key_points, "keypoint_confidence", None)
        if confidence is None:
            confidence = key_points.confidence
        if confidence is None:
            return None
        mask = confidence[0] > self.confidence_threshold
        target = np.array(self.config.vertices, dtype=np.float32)[mask]
        if not points_well_spread(target, self.min_std):
            return None
        try:
            return ViewTransformer(
                source=key_points.xy[0][mask].astype(np.float32),
                target=target)
        except ValueError:
            return None

    def update(self, key_points) -> Optional["ViewTransformer"]:
        """Calibration for this frame, falling back to a recent good one."""
        transformer = self.compute(key_points)
        if transformer is not None:
            self._last, self._age = transformer, 0
            return transformer
        self._age += 1
        return self._last if self._age <= self.max_age else None

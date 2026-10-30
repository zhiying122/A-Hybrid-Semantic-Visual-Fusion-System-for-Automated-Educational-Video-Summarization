import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.hand_tracker import HandTracker


def test_roi_origin_keeps_hand_coordinates_local_to_crop():
    tracker = object.__new__(HandTracker)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    roi, (origin_x, origin_y) = tracker.extract_roi_with_origin(
        frame,
        (320, 240),
    )

    assert roi.shape[:2] == (224, 224)
    assert (320 - origin_x, 240 - origin_y) == (112, 112)


def test_roi_origin_handles_frame_boundaries():
    tracker = object.__new__(HandTracker)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    roi, (origin_x, origin_y) = tracker.extract_roi_with_origin(
        frame,
        (10, 20),
    )

    assert (origin_x, origin_y) == (0, 0)
    assert (10 - origin_x, 20 - origin_y) == (10, 20)
    assert roi.size > 0

"""Image preparation applied before a receipt photograph reaches the OCR engine.

A supplier bill does not arrive as a scan.  It arrives as a phone photograph
taken over a counter: held at an angle, lit from one side, often of a thermal
till roll whose ink has already started to fade.  The detection stage of an OCR
engine is reasonably tolerant of that, but the recognition stage is not, and the
cheapest accuracy available in this whole system comes from fixing the geometry
and the contrast before the model ever sees the pixels.

Each step here is a separate function for two reasons.  The first is that they
are independently testable, which matters because a silent regression in deskew
shows up later as a mysterious drop in line-item recall.  The second is that not
every step helps every image, so the caller is able to compose its own pipeline
rather than being handed one opaque transform.

One deliberate omission is worth stating.  ``adaptive_threshold`` exists and is
tested, but ``prepare_for_ocr`` does not use it.  Hard binarisation suits the
classical pipelines that preceded neural recognisers, and it destroys exactly the
grey-level detail a CRNN uses to separate a faded 8 from a 3.  It is kept for
callers working on near-monochrome text and for the barcode and QR path, where a
clean two-level image genuinely does help.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "to_greyscale",
    "normalise_contrast",
    "estimate_skew_angle",
    "deskew",
    "adaptive_threshold",
    "upscale_if_small",
    "prepare_for_ocr",
]

#: Skew beyond this is not page rotation, it is a mis-detected table rule or a
#: photograph taken sideways.  Rotating by such an angle makes things worse, so
#: an estimate outside the band is discarded rather than trusted.
MAX_SKEW_DEGREES = 15.0

#: Below this short side the recogniser starts guessing at small print.  Receipt
#: photographs cropped from a wider frame are routinely smaller than this.
MIN_SHORT_SIDE = 900


def to_greyscale(image: np.ndarray) -> np.ndarray:
    """Collapse to a single channel.

    Receipt ink is black on white or grey thermal paper, so colour carries no
    information about the text and only triples the work.  Colour is dropped
    first so every later step operates on one plane.
    """
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalise_contrast(grey: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Equalise local contrast with CLAHE.

    A photograph taken under a shop light has a bright band where the light
    falls and a dim corner where the hand shadows the paper.  A global stretch
    cannot fix both ends at once, because the dim corner's ink is darker than the
    bright band's paper.  CLAHE works per tile, so each part of the bill is
    stretched against its own neighbourhood, and the clip limit stops it from
    amplifying the paper grain into fake strokes.
    """
    grey = to_greyscale(grey)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(grey)


def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate page rotation in degrees from the angle of the text lines.

    The reliable signal on a receipt is that printed text forms long horizontal
    runs, and nothing else on the paper does.  So the ink is isolated, dilated
    sideways until the words of one line fuse into a single blob, and the angle
    of each blob's minimum-area rectangle is read off.  The median across blobs
    is used rather than the mean, because one logo or one stray fold produces a
    wildly tilted blob that would drag an average with it.

    A positive angle means the text rises to the right.  Zero is returned when
    there is nothing confidently line-shaped to measure, which is the safe answer
    because the caller then leaves the image alone.
    """
    grey = to_greyscale(image)
    # Invert so ink is bright, which is what the morphology operators expect.
    binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # The kernel is sized from the image width so the same code works on a
    # 400 pixel crop and a 3000 pixel photograph.
    width = max(grey.shape[1] // 30, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 3))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    min_length = grey.shape[1] * 0.15
    for contour in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(w, h), min(w, h)
        if long_side < min_length or short_side < 3:
            continue
        # A text line is far wider than it is tall; anything squarer is a logo,
        # a stamp or a block of the photograph's background.
        if long_side / max(short_side, 1.0) < 4.0:
            continue
        if w < h:
            angle += 90.0
        # OpenCV reports the rectangle angle in [0, 90); fold it into a signed
        # tilt about the horizontal.
        if angle > 45.0:
            angle -= 90.0
        if abs(angle) <= MAX_SKEW_DEGREES:
            angles.append(-angle)

    if len(angles) < 2:
        return 0.0
    return float(np.median(angles))


def deskew(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Rotate the image so the text lines run horizontally.

    Row grouping downstream decides which words belong together by comparing
    their vertical extents.  On a bill tilted by even three degrees the right
    hand amount of one line overlaps the left hand name of the next, and the
    parser then multiplies a quantity by the wrong price.  Straightening the
    page is therefore not cosmetic, it is what makes the row logic valid.

    Rotation is done about the centre with the canvas expanded to fit, so no
    corner of the bill is cropped away, and the new border is filled by
    replicating the edge rather than with black, because a black margin invents
    a strong edge the text detector then reports as a line of print.
    """
    if angle is None:
        angle = estimate_skew_angle(image)
    if abs(angle) < 0.2:
        # Below a fifth of a degree the interpolation blur costs more than the
        # tilt does.
        return image

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def adaptive_threshold(image: np.ndarray, block: int = 31, offset: int = 10) -> np.ndarray:
    """Two-level image using a per-neighbourhood threshold.

    A single global cut-off fails on receipts for the same reason a global
    contrast stretch does: the paper in the lit half is brighter than the ink in
    the shadowed half, so no one value separates them.  The threshold is instead
    computed from each pixel's own neighbourhood.  The block size must be odd and
    large enough to contain both ink and paper, otherwise the inside of a thick
    stroke is judged against other ink and comes back white.
    """
    grey = to_greyscale(image)
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        grey,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        offset,
    )


def upscale_if_small(image: np.ndarray, min_short_side: int = MIN_SHORT_SIDE) -> np.ndarray:
    """Enlarge images whose text is too small for the recogniser to resolve.

    The recognition model reads a fixed-height strip, so a character only a few
    pixels tall is resampled up inside the engine anyway, with no control over
    how.  Doing it here with cubic interpolation gives smoother strokes than the
    engine's own resize, and it is skipped entirely for images that are already
    large, where it would only cost time and memory.
    """
    h, w = image.shape[:2]
    short = min(h, w)
    if short >= min_short_side or short == 0:
        return image
    scale = min(min_short_side / short, 4.0)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def prepare_for_ocr(
    image: np.ndarray,
    *,
    do_deskew: bool = True,
    do_contrast: bool = True,
    do_upscale: bool = True,
) -> np.ndarray:
    """The default pipeline: contrast, then geometry, then size.

    The order is not arbitrary.  Contrast comes first because the skew estimate
    depends on thresholding the ink cleanly, and on a badly lit photograph the
    threshold loses whole lines that the estimate needed.  Upscaling comes last
    so the expensive interpolation runs once, on the final geometry, rather than
    being redone by the rotation.

    The result is a three channel image because the engine wrapper expects one,
    and grey text is written into all three channels rather than being left as a
    single plane so that callers can pass the output straight to code that
    assumes colour.
    """
    grey = normalise_contrast(image) if do_contrast else to_greyscale(image)
    if do_deskew:
        grey = deskew(grey)
    if do_upscale:
        grey = upscale_if_small(grey)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

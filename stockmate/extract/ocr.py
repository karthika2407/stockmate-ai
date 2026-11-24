"""Text recognition for bill photographs, wrapped so the rest of the system
never imports the engine directly.

The engine is RapidOCR on onnxruntime: a detection model, an angle classifier
and a recognition model, all shipped as ONNX files inside the installed package
and all running on the CPU.  That choice is a requirement rather than a
preference.  A shopkeeper photographing a supplier bill is holding a document
with their GSTIN, their purchase prices and their supplier's terms on it, and
sending that to a hosted OCR service would put commercially sensitive paper on
somebody else's machine for no accuracy gain worth having.  Everything here runs
offline with no key, no quota and no per-page cost.

Two practical facts shape the interface.  The models take about half a second to
load and are then reusable for the life of the process, so they are held in a
module level singleton and the first caller is not made to pay for everybody.
And the engine returns loose word boxes, not lines: a receipt row such as a name,
a quantity and two amounts arrives as four unrelated detections.  Reassembling
those into visual rows is the job this module finishes, because every parsing
rule downstream is expressed in terms of a row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["OcrWord", "OcrLine", "OcrResult", "run_ocr", "warmup", "group_into_rows"]


@dataclass(frozen=True)
class OcrWord:
    """One detection as the engine reported it."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]      # x0, y0, x1, y1

    @property
    def centre_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


@dataclass
class OcrLine:
    """One visual row of the bill, rebuilt from the words that sit on it.

    ``text`` is the words joined left to right with single spaces, which is what
    keyword matching reads.  ``words`` is kept because the parser needs to know
    which token a number came from in order to attribute a confidence to the
    field, and to tell a quantity on the left from an amount on the right.
    """

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    centre_y: float
    words: list[OcrWord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
        }


@dataclass
class OcrResult:
    """Everything one image produced, including what it cost."""

    lines: list[OcrLine]
    elapsed_ms: float
    engine: str = "rapidocr_onnxruntime"
    engine_version: str = ""
    image_size: tuple[int, int] = (0, 0)   # width, height

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.confidence for l in self.lines) / len(self.lines)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "line_count": len(self.lines),
            "mean_confidence": round(self.mean_confidence, 4),
            "lines": [l.to_dict() for l in self.lines],
        }


# --------------------------------------------------------------- engine handle

_ENGINE = None
_ENGINE_VERSION = ""


def _engine():
    """Return the shared engine, constructing it on first use.

    Loading three ONNX graphs is the single most expensive thing in the
    extraction path and it is pure start-up cost: the same session serves every
    later image.  Importing the package at module scope would also mean that
    anything merely importing ``stockmate.extract`` paid for onnxruntime, which
    makes the unrelated forecasting tests noticeably slower to collect.
    """
    global _ENGINE, _ENGINE_VERSION
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        try:
            from rapidocr_onnxruntime import __version__ as version
        except Exception:                                  # pragma: no cover
            version = "unknown"
        _ENGINE = RapidOCR()
        _ENGINE_VERSION = str(version)
    return _ENGINE


def warmup() -> float:
    """Load the models and run one tiny image through them.

    Called before any timing is taken.  The first real call otherwise absorbs
    model loading and the lazy allocation onnxruntime does on its first inference,
    which would be charged to whichever bill happened to be photographed first
    and would make a per-receipt latency figure meaningless.
    """
    t0 = time.perf_counter()
    engine = _engine()
    blank = np.full((64, 256, 3), 255, dtype=np.uint8)
    try:
        engine(blank)
    except Exception:
        # A blank image legitimately yields nothing; some versions signal that
        # by raising rather than returning an empty list, and either way the
        # models are now resident, which is the point of this call.
        pass
    return (time.perf_counter() - t0) * 1000.0


# ------------------------------------------------------------- row reassembly


def group_into_rows(words: list[OcrWord], overlap: float = 0.4) -> list[OcrLine]:
    """Group word boxes into visual rows, each ordered left to right.

    Rows are grown greedily down the page.  A word joins the row being built when
    its vertical span overlaps the row's span by more than ``overlap`` of the
    shorter of the two heights.  Comparing spans rather than centre distances is
    what makes this work on receipts, where an amount is often set in a larger
    face than the item name beside it and the two centres can sit several pixels
    apart while the boxes plainly share a line.

    The overlap fraction is the one tuning knob.  Too low and the amount column
    of one line is swept in with the name of the next; too high and a slightly
    raised superscript breaks a row in two.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.centre_y, w.bbox[0]))
    rows: list[list[OcrWord]] = [[ordered[0]]]
    for word in ordered[1:]:
        current = rows[-1]
        top = min(w.bbox[1] for w in current)
        bottom = max(w.bbox[3] for w in current)
        shared = min(bottom, word.bbox[3]) - max(top, word.bbox[1])
        reference = min(bottom - top, word.height) or 1.0
        if shared / reference >= overlap:
            current.append(word)
        else:
            rows.append([word])

    lines: list[OcrLine] = []
    for row in rows:
        row.sort(key=lambda w: w.bbox[0])
        x0 = min(w.bbox[0] for w in row)
        y0 = min(w.bbox[1] for w in row)
        x1 = max(w.bbox[2] for w in row)
        y1 = max(w.bbox[3] for w in row)
        lines.append(
            OcrLine(
                text=" ".join(w.text for w in row).strip(),
                confidence=min(w.confidence for w in row),
                bbox=(x0, y0, x1, y1),
                centre_y=(y0 + y1) / 2.0,
                words=row,
            )
        )
    lines.sort(key=lambda l: l.centre_y)
    return lines


# ------------------------------------------------------------------- main call


def _as_array(image) -> np.ndarray:
    """Accept a numpy array or a PIL image and hand back BGR pixels."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            import cv2

            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    # A PIL image: RGB order has to be flipped, because the detection model was
    # trained on OpenCV's channel order and a swapped image quietly loses recall
    # rather than failing.
    array = np.array(image.convert("RGB"))
    return array[:, :, ::-1].copy()


def run_ocr(image, *, row_overlap: float = 0.4) -> OcrResult:
    """Recognise all text in one bill image and return it as visual rows.

    The elapsed time covers recognition only, not image preparation, so that a
    latency figure can be attributed to the engine rather than to whatever the
    caller chose to do to the pixels first.
    """
    array = _as_array(image)
    height, width = array.shape[:2]

    engine = _engine()
    t0 = time.perf_counter()
    raw = engine(array)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # RapidOCR returns (detections, timing breakdown), and a page with no text
    # at all comes back with detections set to None rather than an empty list.
    detections = raw[0] if isinstance(raw, tuple) else raw
    words: list[OcrWord] = []
    for item in detections or []:
        box, text, score = item[0], item[1], item[2]
        text = str(text).strip()
        if not text:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        words.append(
            OcrWord(
                text=text,
                confidence=float(score),
                bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            )
        )

    return OcrResult(
        lines=group_into_rows(words, overlap=row_overlap),
        elapsed_ms=elapsed_ms,
        engine_version=_ENGINE_VERSION,
        image_size=(width, height),
    )

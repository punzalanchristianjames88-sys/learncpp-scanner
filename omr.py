"""
Answer-sheet reader for the LearnC++ OMR scanner.

Pipeline, in order:

    validate_scan()        is the photo usable at all?
    detect_corner_markers  find the four printed black squares
    order_corner_points    sort them TL, TR, BR, BL
    perspective_correct    warp them onto a standard rectangle
    preprocess_image       grayscale + denoise
    get_bubble_roi         locate each bubble from fixed template geometry
    calculate_fill_ratio   how much of the bubble's INTERIOR is inked
    detect_answer          BLANK / MULTIPLE / AMBIGUOUS / a letter, plus confidence
    detect_layout          which printed layout this is (see the layout bar)
    detect_set_version     the A/B/C column, read the same way
    read_student_id        the Student ID columns (users.id), newer sheets only
    scan_answer_sheet      the whole thing, returning structured data

Everything the sheet's layout implies lives in SHEET (older sheets, Set column
outside the frame) and SHEET_V2 (current sheets, Set and Student ID inside the
frame) below, so recalibrating means editing one block rather than hunting
coordinates through the code.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Debugging
# --------------------------------------------------------------------------
# Set OMR_DEBUG_IMAGES=1 to dump every intermediate stage to uploads/debug/.
DEBUG_MODE = os.getenv('OMR_DEBUG_IMAGES', '').strip() not in ('', '0', 'false', 'False')
DEBUG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads', 'debug')


def _debug_save(name: str, image) -> None:
    if not DEBUG_MODE or image is None:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f'{name}.jpg'), image)
    except Exception:
        pass                      # debugging must never break a scan


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
# Every threshold below can be overridden by an environment variable, so a
# deployment can be re-tuned against its own printer and lighting without
# editing code. The defaults are what this repo was calibrated to.
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, '').strip()
    if raw == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if raw == '':
        return default
    return raw not in ('0', 'false', 'no', 'off')


# --------------------------------------------------------------------------
# Sheet template geometry  (see modules/lessons/final_quiz.php)
# --------------------------------------------------------------------------
@dataclass
class SheetGeometry:
    """Fixed layout of the printed sheet, in fractions rather than pixels."""

    QUESTION_ROWS: int = 10
    QUESTION_COLUMNS: int = 4          # Q1-10, Q11-20, Q21-30, Q31-40
    OPTIONS: tuple = ('a', 'b', 'c', 'd')

    # Standard size the grid is warped onto. 800:710 is the printed frame's
    # aspect ratio (.omr-grid in final_quiz.php), so squares stay square.
    GRID_WIDTH: int = 1071
    GRID_HEIGHT: int = 951

    # The registration marks are printed this far outside the grid, as a
    # fraction of the grid's own width/height (.reg offsets, -6% / 106%).
    MARKER_MARGIN: float = 0.06

    # Each question row is five equal columns: [number][A][B][C][D].
    COLUMN_POSITIONS: tuple = (0.30, 0.50, 0.70, 0.90)   # of cell width
    ROW_POSITION: float = 0.50                            # of cell height

    # Printed bubble size, as a fraction of one question cell.
    BUBBLE_W_FRAC: float = 0.131       # 5mm circle in a 38.25mm-wide cell
    BUBBLE_H_FRAC: float = 0.368       # 5mm circle in a 13.58mm-tall row

    # Read only the inside of the circle so the printed outline is never
    # counted as a student's mark, and take the paper reference from a ring
    # just outside it so shadows across the row cancel out.
    INNER_RADIUS_FRAC: float = 0.62
    PAPER_RING_INNER: float = 1.25
    PAPER_RING_OUTER: float = 1.70

    # Set Version column (.zg-set), which sits OUTSIDE the marker frame.
    # Offsets are relative to the grid's top-left, in grid widths/heights.
    # These are derived from the print CSS and are the values most likely to
    # need calibrating against a real printout.
    SET_VERSION_OPTIONS: tuple = ('a', 'b', 'c')
    SET_CENTER_X: float = -0.173       # midway across the 15mm set column
    SET_FIRST_CENTER_Y: float = -0.070
    SET_PITCH_Y: float = 0.038
    SET_BUBBLE_W_FRAC: float = 0.026   # 4mm circle, in grid widths
    SET_BUBBLE_H_FRAC: float = 0.029

    # Canvas padding around the grid, so the set column and the markers
    # themselves land inside the warped image.
    PAD_LEFT: float = 0.35
    PAD_TOP: float = 0.14
    PAD_RIGHT: float = 0.12
    PAD_BOTTOM: float = 0.12


SHEET = SheetGeometry()


@dataclass
class SheetLayoutV2:
    """Current sheets: the Set and Student ID bubbles sit INSIDE the frame.

    Every position is a fraction of the marker frame -- the rectangle whose
    corners are the four marker centres -- so it maps straight onto the fixed
    canvas. The same percentages place the blocks in .omr-frame in
    final_quiz.php; change one and change the other. Boxes are
    (left, top, width, height), split into equal cells with one bubble
    centred in each.
    """

    # The frame as printed on A4, used only to turn bubble sizes into pixels.
    PRINT_FRAME_MM: tuple = (184.8, 164.0)

    # 4 question columns x 10 rows; each cell is [number 0.6][A][B][C][D].
    ANSWER_BOX: tuple = (0.255, 0.105, 0.695, 0.820)
    ANSWER_COLUMNS: int = 4
    ANSWER_ROWS: int = 10
    ANSWER_OPTION_X: tuple = (1.1 / 4.6, 2.1 / 4.6, 3.1 / 4.6, 4.1 / 4.6)   # of a cell
    ANSWER_BUBBLE_MM: float = 5.0

    SET_BOX: tuple = (0.046, 0.150, 0.184, 0.050)        # one row: A, B, C
    SET_BUBBLE_MM: float = 5.0

    # users.id, first digit in the first column; the rows are the digits 0-9.
    ID_BOX: tuple = (0.070, 0.330, 0.160, 0.550)
    ID_COLUMNS: int = 5
    ID_DIGITS: tuple = tuple('0123456789')
    ID_BUBBLE_MM: float = 4.0      # 3.6mm inside the outline, 4.4mm across it

    # A solid bar printed only on this layout, on a spot that is blank paper
    # on older sheets -- how the reader tells the two apart.
    LAYOUT_MARK: tuple = (0.460, 0.948, 0.080, 0.020)


SHEET_V2 = SheetLayoutV2()


# --------------------------------------------------------------------------
# Scoring thresholds  — the knobs worth calibrating
# --------------------------------------------------------------------------
MARK_THRESHOLD = _env_float('OMR_MARK_THRESHOLD', 0.30)      # inked fraction of a bubble's interior => filled
AMBIGUOUS_MARGIN = _env_float('OMR_AMBIGUOUS_MARGIN', 0.12)  # best must beat second by this much
MULTIPLE_THRESHOLD = _env_float('OMR_MULTIPLE_THRESHOLD', 0.30)  # a second bubble this filled => MULTIPLE

# How much darker than its surrounding paper a pixel must be to count as ink.
# Measured on a real blank sheet, an unmarked bubble's interior scores 0.000 at
# every setting from 0.28 down to 0.10 -- the circular inner mask keeps the
# printed outline out, so there is simply nothing dark inside. That headroom is
# what lets this sit low enough to catch a faint pencil mark: at 0.28 a light
# grey mark was still brighter than the ink cut-off and read as BLANK.
INK_DROP = _env_float('OMR_INK_DROP', 0.18)

# Image-quality gates. These reject only what is unambiguously unusable.
# Shortest side. Set from what actually still grades: a 684x366 crop of a real
# sheet reads correctly, so anything above ~300 is workable. A higher floor was
# rejecting readable photos.
MIN_RESOLUTION = _env_float('OMR_MIN_RESOLUTION', 300)
MIN_MEAN_LEVEL = _env_float('OMR_MIN_MEAN_LEVEL', 35)    # too dark
MAX_MEAN_LEVEL = _env_float('OMR_MAX_MEAN_LEVEL', 245)   # blown out

# Blur is reported but never blocks. Real phone photos of the sheet measure
# 400-1800 on variance-of-Laplacian, so any threshold high enough to catch a
# genuinely soft photo also throws away readable ones. Marker detection is the
# real quality gate: if the sheet is too blurred to read, the four squares are
# not found and the user gets a clear message instead of invented answers.
SHARPNESS_WARN = _env_float('OMR_SHARPNESS_WARN', 25.0)

BLANK, MULTIPLE, AMBIGUOUS = 'BLANK', 'MULTIPLE', 'AMBIGUOUS'


# --------------------------------------------------------------------------
# 1. Corner markers
# --------------------------------------------------------------------------
MARKER_MIN_AREA_FRAC = 0.00002
MARKER_MAX_AREA_FRAC = 0.03
MARKER_SQUARENESS = 0.75     # short/long side of the fitted rectangle
MARKER_FILL = 0.82           # a square fills ~1.0 of it; a circle only 0.785
MARKER_SIDE_RATIO = 1.5      # opposite sides of the quad stay similar

# The OPEN rescue below runs on weaker evidence than the clean pass, so it asks
# for a better-formed quad. Measured on real photos: five genuine rescued sheets
# all sit at 1.02, while a photo of a screen whose shaded bubbles impersonated
# markers came out at 1.46. Rectangularity is the right test here because it does
# not depend on the printed layout the way an aspect ratio does.
MARKER_RESCUE_SIDE_RATIO = 1.25

# The four marks are the grid scaled uniformly, so a correctly framed sheet
# photographs at close to the grid's own aspect however far away the phone is.
# Measured across 115 real photos where markers were found: the 24 inside this
# band read 30.5 of 40 on average, while the 91 outside it never got past 18 --
# every one of those was a different sheet layout or four false markers. The
# check turns a silently wrong score into an honest retake.
MARKER_ASPECT_TOLERANCE = _env_float('OMR_ASPECT_TOLERANCE', 1.75)

# A printed marker sits alone on paper; a shaded bubble is hemmed in by its
# neighbours, the row box and the question number. This measures the SHARE of
# the surrounding ring that is clearly lighter than the blob.
#
# Measured on real photos: the four markers score 1.00 (paper on every side),
# while every candidate on an older unmarked sheet scores 0.00. An earlier
# version tested the ring's standard deviation instead, and that quietly broke
# real scans -- the markers sit near the page edge, so the ring caught the dark
# desk behind the paper, the deviation spiked, and all four were thrown out.
# A share is robust to part of the ring falling off the page; a deviation is not.
# Swept against real photos: at contrast 3.0 all marked sheets are still found
# while false matches on older unmarked sheets halve. The share stays at 0.70
# because 0.90 is past the cliff -- there it starts losing real markers.
# How much brighter than the blob a ring pixel must be to count as paper.
#
# This was once a multiple of the blob's own level (ring > core x 3.0), which
# silently assumed the squares print near-black. They do not: a real printout
# measures 72-95 on white paper of 210-217, so the rule demanded a ring of
# 216-284 -- above 255, and therefore impossible, at two of the four corners.
# Every marker was rejected and every photo of a real sheet was refused, while
# the pure-black test render (core 1.0) sailed through. A gap is the right
# shape of test: what matters is that paper stands clear of the mark, not that
# the mark is almost black.
MARKER_RING_MIN_GAP = _env_float('OMR_RING_MIN_GAP', 40.0)    # absolute levels
MARKER_RING_GAP_FRAC = _env_float('OMR_RING_GAP_FRAC', 0.30)  # of the headroom above the blob
MARKER_RING_MIN_SHARE = _env_float('OMR_RING_MIN_SHARE', 0.70)
MARKER_RING_OUTER = 1.50


def _marker_is_isolated(gray, cx, cy, size) -> bool:
    """True when the blob is ringed by paper rather than by other printing."""
    r_in, r_out = size * 0.85, size * MARKER_RING_OUTER
    h, w = gray.shape[:2]
    # Clamp to the image: a marker near the edge of the frame is still valid.
    x0, y0 = max(0, int(cx - r_out)), max(0, int(cy - r_out))
    x1, y1 = min(w, int(cx + r_out)), min(h, int(cy + r_out))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cheb = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    ring = patch[(cheb >= r_in) & (cheb <= r_out)]
    core = patch[cheb <= size * 0.35]
    if ring.size < 24 or core.size < 4:
        return False

    core_mean = float(np.mean(core))
    # Scale the gap to the headroom the mark leaves: a near-black square gets a
    # wide margin, a grey one still gets a workable margin instead of an
    # unreachable threshold. The share test below is what keeps a shaded bubble
    # out -- its ring is crowded with the row box and its neighbours, not paper.
    gap = max(MARKER_RING_MIN_GAP, MARKER_RING_GAP_FRAC * (255.0 - core_mean))
    bright = float(np.count_nonzero(ring > core_mean + gap))
    return (bright / ring.size) >= MARKER_RING_MIN_SHARE


def _marker_candidates(gray, binary):
    """Every blob in `binary` that looks like a printed registration square."""
    # RETR_LIST, not RETR_EXTERNAL: the paper's edge against the desk forms a
    # closed ring around everything, and the markers sit inside it.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    img_area = float(h * w)

    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MARKER_MIN_AREA_FRAC * img_area <= area <= MARKER_MAX_AREA_FRAC * img_area):
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)   # rotation-invariant
        if rw < 1 or rh < 1:
            continue
        if min(rw, rh) / max(rw, rh) < MARKER_SQUARENESS:
            continue
        if area < MARKER_FILL * rw * rh:
            continue
        if not _marker_is_isolated(gray, cx, cy, max(rw, rh)):
            continue
        found.append((cx, cy, area))
    return found


def _quad_from_candidates(found, w, h, side_ratio=MARKER_SIDE_RATIO):
    """Turn marker candidates into an ordered quad, or None."""
    if len(found) < 4:
        return None

    # The four markers print identically, so keep the largest group of
    # similar-sized squares and try each group until one forms a sane quad.
    found.sort(key=lambda m: -m[2])
    seen = set()
    for m in found:
        peers = [n for n in found if 0.6 * m[2] <= n[2] <= 1.7 * m[2]]
        if len(peers) < 4:
            continue
        key = tuple(sorted((round(p[0], 1), round(p[1], 1)) for p in peers))
        if key in seen:
            continue
        seen.add(key)
        quad = order_corner_points(np.array([[p[0], p[1]] for p in peers], dtype='float32'))
        if _quad_is_plausible(quad, w, h, side_ratio):
            return quad
    return None


def detect_corner_markers(image):
    """Locate the four black squares. Returns 4x2 float32 or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Adaptive, so one bright corner or a shadow never loses a marker.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _debug_save('02_marker_threshold', binary)

    h, w = gray.shape[:2]
    clean = _marker_candidates(gray, binary)
    quad = _quad_from_candidates(clean, w, h)
    if quad is not None:
        return quad

    # Only a sheet that showed some real markers gets the rescue below. On a
    # sheet that never had them there is nothing to corroborate, and inventing
    # a quad there is the one outcome worse than falling back to the legacy
    # reader: the grid lands on nothing and every answer is fiction.
    if len(clean) < 2:
        return None

    # Printing that runs close to a marker welds itself onto it: the sheet's
    # rotated instruction text sits a few pixels from the two TOP squares, the
    # threshold joins them, and the merged blob is a 32x59 bar -- far from
    # square, so it is thrown out. Two markers left, no quad, and the whole
    # sheet silently drops to the legacy reader, which then misreads bubbles
    # that are plainly shaded.
    #
    # An OPEN erases strokes a few pixels wide while a solid ~30px square only
    # loses and regains its border, so the squares come back cleanly. It runs
    # only as a retry: sheets that already resolve never touch it, and nothing
    # that works today can be eroded away by it.
    #
    # The same OPEN also frees a shaded bubble from the row box that normally
    # disqualifies it. A blurred bubble and a blurred square are then not
    # separable by shape or darkness -- measured across real photos both land
    # at fill 0.82-0.89 over a core of 12-48 grey -- so the result is only
    # trusted when it is exactly four squares AND it still contains the ones
    # the clean pass was already sure of.
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _debug_save('02b_marker_threshold_opened', opened)

    rescued = _marker_candidates(gray, opened)
    if len(rescued) != 4:
        return None

    for cx, cy, area in clean:
        tol = max(6.0, 0.5 * float(np.sqrt(max(area, 1.0))))
        if not any(abs(cx - rx) <= tol and abs(cy - ry) <= tol for rx, ry, _ in rescued):
            return None

    return _quad_from_candidates(rescued, w, h, side_ratio=MARKER_RESCUE_SIDE_RATIO)


def order_corner_points(pts):
    """Sort points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype='float32')
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)],   # TL: smallest x+y
                     pts[np.argmin(d)],   # TR: smallest y-x
                     pts[np.argmax(s)],   # BR: largest x+y
                     pts[np.argmax(d)]],  # BL: largest y-x
                    dtype='float32')


def _quad_is_plausible(quad, img_w, img_h, side_ratio=MARKER_SIDE_RATIO) -> bool:
    if len({tuple(np.round(p, 1)) for p in quad}) != 4:
        return False
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    if max(top, bottom) < 0.12 * img_w or max(left, right) < 0.08 * img_h:
        return False
    # Four markers photographed off-angle still form a near-rectangle.
    if max(top, bottom) / max(min(top, bottom), 1.0) > side_ratio:
        return False
    if max(left, right) / max(min(left, right), 1.0) > side_ratio:
        return False
    return bool(cv2.isContourConvex(quad.astype(np.int32)))


# --------------------------------------------------------------------------
# 2. Perspective correction
# --------------------------------------------------------------------------
def _canvas_layout(g: SheetGeometry = SHEET):
    """Where the grid sits on the warped canvas, and how big the canvas is."""
    ox = g.PAD_LEFT * g.GRID_WIDTH
    oy = g.PAD_TOP * g.GRID_HEIGHT
    width = int(ox + g.GRID_WIDTH * (1.0 + g.PAD_RIGHT))
    height = int(oy + g.GRID_HEIGHT * (1.0 + g.PAD_BOTTOM))
    return ox, oy, width, height


def perspective_correct(image, quad, g: SheetGeometry = SHEET):
    """Warp so the markers land on their exact template positions.

    The canvas is larger than the grid so the Set Version column, which is
    printed outside the marker frame, is still inside the corrected image.
    """
    ox, oy, cw, ch = _canvas_layout(g)
    m = g.MARKER_MARGIN
    dst = np.array([
        [ox - m * g.GRID_WIDTH,             oy - m * g.GRID_HEIGHT],
        [ox + (1 + m) * g.GRID_WIDTH,       oy - m * g.GRID_HEIGHT],
        [ox + (1 + m) * g.GRID_WIDTH,       oy + (1 + m) * g.GRID_HEIGHT],
        [ox - m * g.GRID_WIDTH,             oy + (1 + m) * g.GRID_HEIGHT],
    ], dtype='float32')

    matrix = cv2.getPerspectiveTransform(quad.astype('float32'), dst)
    return cv2.warpPerspective(image, matrix, (cw, ch), borderValue=(255, 255, 255))


# --------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------
def preprocess_image(warped):
    """Grayscale + gentle denoise. No global threshold: every bubble is judged
    against the paper beside it, which handles uneven lighting far better than
    one cut-off applied to the whole sheet."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _debug_save('04_grayscale', gray)
    if DEBUG_MODE:
        _debug_save('05_adaptive_threshold',
                    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 41, 12))
    return gray


# --------------------------------------------------------------------------
# 4/5. Bubble ROIs and fill ratio
# --------------------------------------------------------------------------
def get_bubble_roi(question: int, option_index: int, g: SheetGeometry = SHEET):
    """Centre and radii of one answer bubble on the warped canvas."""
    ox, oy, _, _ = _canvas_layout(g)
    col = (question - 1) // g.QUESTION_ROWS
    row = (question - 1) % g.QUESTION_ROWS
    cell_w = g.GRID_WIDTH / g.QUESTION_COLUMNS
    cell_h = g.GRID_HEIGHT / g.QUESTION_ROWS

    cx = ox + col * cell_w + g.COLUMN_POSITIONS[option_index] * cell_w
    cy = oy + row * cell_h + g.ROW_POSITION * cell_h
    radius = 0.5 * min(g.BUBBLE_W_FRAC * cell_w, g.BUBBLE_H_FRAC * cell_h)
    return cx, cy, radius


def calculate_fill_ratio(gray, cx, cy, radius, g: SheetGeometry = SHEET):
    """Inked fraction of a bubble's interior.

    A circular mask at INNER_RADIUS_FRAC keeps the printed outline out of the
    measurement, and the paper level comes from a ring just outside the circle
    so a shadow gradient across the row cannot masquerade as pencil.
    """
    r_out = radius * g.PAPER_RING_OUTER
    x0, y0 = int(cx - r_out), int(cy - r_out)
    x1, y1 = int(cx + r_out), int(cy + r_out)
    h, w = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return 0.0, 255.0

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = patch[dist <= radius * g.INNER_RADIUS_FRAC]
    ring = patch[(dist >= radius * g.PAPER_RING_INNER) &
                 (dist <= radius * g.PAPER_RING_OUTER)]
    if inner.size < 8 or ring.size < 8:
        return 0.0, 255.0

    paper = float(np.median(ring))
    if paper < 1.0:
        return 0.0, 1.0
    ink_level = paper * (1.0 - INK_DROP)
    return float(np.count_nonzero(inner < ink_level)) / inner.size, paper


# --------------------------------------------------------------------------
# 6/7/8. Deciding an answer
# --------------------------------------------------------------------------
def detect_answer(scores, options=SHEET.OPTIONS):
    """Turn four fill ratios into a verdict plus a confidence.

    Never guesses: a row of near-empty bubbles is BLANK, two strong marks are
    MULTIPLE, and a narrow win is AMBIGUOUS.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best_i, second_i = order[0], order[1]
    best, second = scores[best_i], scores[second_i]

    # Confidence = how well the winner separates from the runner-up, so it
    # reads as trust in a single usable answer. Two equally filled bubbles
    # score 0, not 1: MULTIPLE is a refusal to answer, not a certainty.
    separation = round(float(min(1.0, max(0.0, (best - second) / max(best, 1e-6)))), 3)

    if best < MARK_THRESHOLD:
        # Nothing is filled. Here the confidence is in the verdict BLANK, and
        # it is high precisely when every bubble is far below the threshold.
        return BLANK, round(float(min(1.0, 1.0 - best / MARK_THRESHOLD)), 3)

    if second >= MULTIPLE_THRESHOLD:
        return MULTIPLE, separation

    if (best - second) < AMBIGUOUS_MARGIN:
        return AMBIGUOUS, separation

    return options[best_i], separation


# --------------------------------------------------------------------------
# 8b. Fixed bubble grid
# --------------------------------------------------------------------------
# The four markers are warped straight onto the corners of a canvas of exactly
# this size, so every bubble lands on the same pixel on every sheet, every time.
# That makes the 160 coordinates below constants rather than something re-derived
# per photo, and it is what the reader actually looks up.
FIXED_CANVAS_WIDTH = 1536
FIXED_CANVAS_HEIGHT = 1345


def _build_fixed_bubble_coords(g: SheetGeometry = SHEET,
                               canvas_w: int = FIXED_CANVAS_WIDTH,
                               canvas_h: int = FIXED_CANVAS_HEIGHT):
    """Materialise the 40 x 4 bubble centres for the fixed canvas.

    The markers sit MARKER_MARGIN outside the grid on each side, so the canvas
    (the marker frame) is the grid enlarged by 2 x that margin. Undo it once
    here and every bubble follows from the printed layout.

    Returns (coords, radius) where coords[q][i] is the (cx, cy) of question q's
    option i -- q is 1-based, i indexes SheetGeometry.OPTIONS.
    """
    m = g.MARKER_MARGIN
    span = 1.0 + 2.0 * m

    grid_x0 = canvas_w * (m / span)
    grid_y0 = canvas_h * (m / span)
    grid_w = canvas_w / span
    grid_h = canvas_h / span

    cell_w = grid_w / g.QUESTION_COLUMNS
    cell_h = grid_h / g.QUESTION_ROWS
    radius = 0.5 * min(g.BUBBLE_W_FRAC * cell_w, g.BUBBLE_H_FRAC * cell_h)

    coords = {}
    for q in range(1, g.QUESTION_ROWS * g.QUESTION_COLUMNS + 1):
        col = (q - 1) // g.QUESTION_ROWS
        row = (q - 1) % g.QUESTION_ROWS
        cy = grid_y0 + row * cell_h + g.ROW_POSITION * cell_h
        coords[q] = tuple(
            (round(grid_x0 + col * cell_w + g.COLUMN_POSITIONS[i] * cell_w, 2), round(cy, 2))
            for i in range(len(g.OPTIONS))
        )
    return coords, round(radius, 2)


# 40 questions x 4 options = 160 fixed centres, e.g. Q1 A = (185.14, 132.1).
FIXED_BUBBLE_COORDS, FIXED_BUBBLE_RADIUS = _build_fixed_bubble_coords()
FIXED_BUBBLE_COUNT = sum(len(v) for v in FIXED_BUBBLE_COORDS.values())


def _v2_point(box, cols, rows, col, row, cell_x=0.5,
              canvas_w: int = FIXED_CANVAS_WIDTH, canvas_h: int = FIXED_CANVAS_HEIGHT):
    """Canvas position of a point in cell (col, row) of a SheetLayoutV2 box."""
    left, top, width, height = box
    fx = left + width * (col + cell_x) / cols
    fy = top + height * (row + 0.5) / rows
    return round(fx * (canvas_w - 1), 2), round(fy * (canvas_h - 1), 2)


def _v2_radius(diameter_mm, layout: SheetLayoutV2 = SHEET_V2,
               canvas_w: int = FIXED_CANVAS_WIDTH, canvas_h: int = FIXED_CANVAS_HEIGHT):
    w_mm, h_mm = layout.PRINT_FRAME_MM
    return round(0.5 * min(diameter_mm / w_mm * (canvas_w - 1),
                           diameter_mm / h_mm * (canvas_h - 1)), 2)


def _build_v2_coords(layout: SheetLayoutV2 = SHEET_V2):
    """Answer, Set and Student ID centres for current sheets, on the fixed canvas.

    Returns (answers, set_points, id_points): answers[q][i] as for the older
    sheet, set_points[i] for A/B/C, id_points[column][digit].
    """
    lay = layout
    answers = {}
    for q in range(1, lay.ANSWER_COLUMNS * lay.ANSWER_ROWS + 1):
        col, row = (q - 1) // lay.ANSWER_ROWS, (q - 1) % lay.ANSWER_ROWS
        answers[q] = tuple(_v2_point(lay.ANSWER_BOX, lay.ANSWER_COLUMNS, lay.ANSWER_ROWS,
                                     col, row, x) for x in lay.ANSWER_OPTION_X)
    set_points = tuple(_v2_point(lay.SET_BOX, 3, 1, i, 0) for i in range(3))
    id_points = tuple(tuple(_v2_point(lay.ID_BOX, lay.ID_COLUMNS, len(lay.ID_DIGITS), c, d)
                            for d in range(len(lay.ID_DIGITS)))
                      for c in range(lay.ID_COLUMNS))
    return answers, set_points, id_points


V2_ANSWER_COORDS, V2_SET_COORDS, V2_ID_COORDS = _build_v2_coords()
V2_ANSWER_RADIUS = _v2_radius(SHEET_V2.ANSWER_BUBBLE_MM)
V2_SET_RADIUS = _v2_radius(SHEET_V2.SET_BUBBLE_MM)
V2_ID_RADIUS = _v2_radius(SHEET_V2.ID_BUBBLE_MM)


def sheet_aspect_factor(quad, g: SheetGeometry = SHEET) -> float:
    """How many times wider the detected frame is than the printed one.

    1.0 is a perfect match; 2.0 means the quad is twice as wide, relative to its
    height, as the sheet this reader is built for.
    """
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    measured = ((top + bottom) / 2.0) / max((left + right) / 2.0, 1.0)
    return measured / (g.GRID_WIDTH / g.GRID_HEIGHT)


def check_sheet_shape(quad, g: SheetGeometry = SHEET):
    """(ok, message, factor) -- is this the sheet the coordinates describe?"""
    factor = sheet_aspect_factor(quad, g)
    lo, hi = 1.0 / MARKER_ASPECT_TOLERANCE, MARKER_ASPECT_TOLERANCE
    if lo <= factor <= hi:
        return True, '', factor
    shape = 'wider' if factor > 1.0 else 'taller'
    return False, (
        'The four squares were found, but the sheet between them is '
        f'{shape} than the LearnC++ answer sheet ({factor:.2f}x). This is '
        'either a different answer sheet or the corners were mistaken for '
        'something else, so it was not graded. Print the answer sheet from '
        'the quiz page and photograph the whole of it.'
    ), factor


def perspective_correct_fixed(image, quad,
                              canvas_w: int = FIXED_CANVAS_WIDTH,
                              canvas_h: int = FIXED_CANVAS_HEIGHT):
    """Warp the marker quad onto the whole fixed canvas, corner to corner."""
    dst = np.array([[0, 0],
                    [canvas_w - 1, 0],
                    [canvas_w - 1, canvas_h - 1],
                    [0, canvas_h - 1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(quad.astype('float32'), dst)
    return cv2.warpPerspective(image, matrix, (canvas_w, canvas_h),
                               borderValue=(255, 255, 255))


def read_fixed_grid(image, quad, total_questions: int = 40, g: SheetGeometry = SHEET):
    """perspective correct -> fixed canvas -> 160 fixed coords -> answers 1-40.

    Grading is unchanged: each bubble is still measured against the paper in the
    ring around it and judged by detect_answer. Only where to look is fixed.
    """
    warped = perspective_correct_fixed(image, quad)
    _debug_save('03c_fixed_canvas', warped)
    gray = preprocess_image(warped)
    answers, confidence, scores = _read_answers(gray, FIXED_BUBBLE_COORDS, FIXED_BUBBLE_RADIUS,
                                                total_questions, g)
    return answers, confidence, scores, warped


def _fill_ratios(gray, points, radius, g: SheetGeometry = SHEET):
    return [round(float(calculate_fill_ratio(gray, cx, cy, radius, g)[0]), 3)
            for cx, cy in points]


def _read_answers(gray, coords, radius, total_questions: int, g: SheetGeometry = SHEET):
    """Verdict, confidence and fill ratios for each question in a coordinate table."""
    answers, confidence, scores = {}, {}, {}
    for q in range(1, total_questions + 1):
        centres = coords.get(q)
        if centres is None:
            continue
        ratios = _fill_ratios(gray, centres, radius, g)
        verdict, conf = detect_answer(ratios, g.OPTIONS)
        answers[q] = verdict
        confidence[q] = conf
        scores[q] = ratios
    return answers, confidence, scores


# --------------------------------------------------------------------------
# 9. Set Version
# --------------------------------------------------------------------------
def detect_set_version(gray, g: SheetGeometry = SHEET):
    """Read the A/B/C column. Returns (letter|None, confidence, scores)."""
    ox, oy, _, _ = _canvas_layout(g)
    radius = 0.5 * min(g.SET_BUBBLE_W_FRAC * g.GRID_WIDTH,
                       g.SET_BUBBLE_H_FRAC * g.GRID_HEIGHT)
    scores = []
    for i in range(len(g.SET_VERSION_OPTIONS)):
        cx = ox + g.SET_CENTER_X * g.GRID_WIDTH
        cy = oy + (g.SET_FIRST_CENTER_Y + i * g.SET_PITCH_Y) * g.GRID_HEIGHT
        ratio, _ = calculate_fill_ratio(gray, cx, cy, radius, g)
        scores.append(ratio)

    verdict, confidence = detect_answer(scores, options=g.SET_VERSION_OPTIONS)
    if verdict in (BLANK, MULTIPLE, AMBIGUOUS):
        # The column sits outside the marker frame, so its position is
        # extrapolated. Rather than risk grading against the wrong key, say
        # nothing -- the caller (process.php) treats an unreadable set the
        # same as any other unreadable bubble: it scores as wrong.
        return None, confidence, scores
    return verdict, confidence, scores


def read_set_v2(gray, g: SheetGeometry = SHEET):
    """The A/B/C row of a current sheet, read off the fixed canvas."""
    scores = _fill_ratios(gray, V2_SET_COORDS, V2_SET_RADIUS, g)
    verdict, confidence = detect_answer(scores, options=g.SET_VERSION_OPTIONS)
    if verdict in (BLANK, MULTIPLE, AMBIGUOUS):
        return None, confidence, scores
    return verdict, confidence, scores


# --------------------------------------------------------------------------
# 10. Student ID (current sheets)
# --------------------------------------------------------------------------
def read_student_id(gray, layout: SheetLayoutV2 = SHEET_V2, g: SheetGeometry = SHEET):
    """users.id from the Student ID columns.

    Returns (id string | None, verdict per column, fill ratios per column).
    Digits run from the first column and the first blank column ends the
    number. Anything else -- two marks in a column, a digit after a blank
    column, a leading zero, nothing at all -- is None rather than a guess.
    """
    columns, scores = [], []
    for points in V2_ID_COORDS:
        ratios = _fill_ratios(gray, points, V2_ID_RADIUS, g)
        verdict, _ = detect_answer(ratios, options=layout.ID_DIGITS)
        columns.append(verdict)
        scores.append(ratios)

    digits = []
    for i, verdict in enumerate(columns):
        if verdict == BLANK:
            if any(v != BLANK for v in columns[i + 1:]):
                return None, columns, scores
            break
        if verdict not in layout.ID_DIGITS:
            return None, columns, scores
        digits.append(verdict)
    if not digits or digits[0] == '0':
        return None, columns, scores
    return ''.join(digits), columns, scores


# --------------------------------------------------------------------------
# 11. Which layout
# --------------------------------------------------------------------------
# The layout bar is solid ink on a current sheet and blank paper on an older
# one. Anything in between -- glare across it, a scribble -- is trusted
# neither way: reading one layout's bubbles off the other would grade every
# answer against the wrong spot.
LAYOUT_MARK_INK = 0.45
LAYOUT_MARK_BLANK = 0.10


def detect_layout(gray, layout: SheetLayoutV2 = SHEET_V2):
    """('v2' | 'v1' | None, inked share of the bar), from the fixed canvas."""
    h, w = gray.shape[:2]
    left, top, width, height = layout.LAYOUT_MARK

    def region(fx0, fx1, fy0, fy1):
        return gray[int(fy0 * (h - 1)):int(fy1 * (h - 1)) + 1,
                    int(fx0 * (w - 1)):int(fx1 * (w - 1)) + 1]

    # The bar's middle, inset so a warp a few pixels off still lands on it,
    # judged against the paper just above and below it.
    core = region(left + 0.15 * width, left + 0.85 * width,
                  top + 0.25 * height, top + 0.75 * height)
    paper = np.concatenate([
        region(left, left + width, top - 0.025, top - 0.008).ravel(),
        region(left, left + width, top + height + 0.008, top + height + 0.025).ravel()])
    if core.size == 0 or paper.size == 0:
        return None, 0.0

    share = float(np.count_nonzero(core < 0.6 * float(np.median(paper)))) / core.size
    if share >= LAYOUT_MARK_INK:
        return 'v2', share
    if share <= LAYOUT_MARK_BLANK:
        return 'v1', share
    return None, share


# --------------------------------------------------------------------------
# 13. Image quality
# --------------------------------------------------------------------------
def validate_scan(image):
    """Return (ok, message). Refuse rather than return invented answers."""
    if image is None or image.size == 0:
        return False, 'Could not read the image.'

    h, w = image.shape[:2]
    if min(h, w) < MIN_RESOLUTION:
        return False, ('The photo is too small to read reliably. '
                       'Move closer and fill the frame with the answer sheet.')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_level = float(np.mean(gray))
    if mean_level < MIN_MEAN_LEVEL:
        return False, 'The photo is too dark. Add more light and try again.'
    if mean_level > MAX_MEAN_LEVEL:
        return False, 'The photo is washed out. Reduce glare and try again.'

    return True, ''


def measure_sharpness(image) -> float:
    """Variance of Laplacian. Reported for diagnostics, never used to reject."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# --------------------------------------------------------------------------
# Debug overlay
# --------------------------------------------------------------------------
def _draw_debug(warped, answers, g: SheetGeometry = SHEET):
    canvas = warped.copy()
    for q in range(1, g.QUESTION_ROWS * g.QUESTION_COLUMNS + 1):
        verdict = answers.get(q, BLANK)
        for i in range(len(g.OPTIONS)):
            cx, cy, r = get_bubble_roi(q, i, g)
            picked = (verdict == g.OPTIONS[i])
            colour = (0, 170, 0) if picked else (200, 160, 60)
            cv2.circle(canvas, (int(cx), int(cy)), int(r), colour, 2)
            cv2.circle(canvas, (int(cx), int(cy)),
                       int(r * g.INNER_RADIUS_FRAC), (255, 120, 0), 1)
        cx0, cy0, r0 = get_bubble_roi(q, 0, g)
        label = verdict if verdict in (BLANK, MULTIPLE, AMBIGUOUS) else verdict.upper()
        cv2.putText(canvas, f'Q{q}:{label}', (int(cx0 - r0 * 3.4), int(cy0 - r0 * 1.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 200), 1, cv2.LINE_AA)
    return canvas


def _draw_debug_v2(fixed, answers, set_version, id_columns, layout: SheetLayoutV2 = SHEET_V2):
    canvas = fixed.copy()

    def rings(points, radius, picked):
        for (cx, cy), hit in zip(points, picked):
            cv2.circle(canvas, (int(cx), int(cy)), int(radius),
                       (0, 170, 0) if hit else (200, 160, 60), 2)
            cv2.circle(canvas, (int(cx), int(cy)),
                       int(radius * SHEET.INNER_RADIUS_FRAC), (255, 120, 0), 1)

    for q, centres in V2_ANSWER_COORDS.items():
        rings(centres, V2_ANSWER_RADIUS, [answers.get(q) == o for o in SHEET.OPTIONS])
    rings(V2_SET_COORDS, V2_SET_RADIUS, [set_version == o for o in SHEET.SET_VERSION_OPTIONS])
    for col, points in enumerate(V2_ID_COORDS):
        verdict = id_columns[col] if col < len(id_columns) else None
        rings(points, V2_ID_RADIUS, [verdict == d for d in layout.ID_DIGITS])
    return canvas


# --------------------------------------------------------------------------
# 16. Public entry point
# --------------------------------------------------------------------------
def scan_answer_sheet(image, total_questions: int = 40, g: SheetGeometry = SHEET,
                      quad=None):
    """Read a whole sheet.

    Returns:
        {
          "success": bool,
          "error": str,                     # only when success is False
          "layout": "v2" | "v1",            # current sheet / older sheet
          "set_version": "a" | None,
          "student_id": "125" | None,       # users.id, current sheets only
          "answers":    {1: "b", 3: "BLANK", ...},
          "confidence": {1: 0.94, ...},
          "scores":     {1: [0.08, 0.12, 0.86, 0.10], ...},
        }
    """
    _debug_save('01_original', image)

    ok, message = validate_scan(image)
    if not ok:
        return {'success': False, 'error': message, 'set_version': None,
                'answers': {}, 'confidence': {}, 'scores': {}}

    if quad is None:
        quad = detect_corner_markers(image)
    if quad is None:
        return {'success': False,
                'error': ('Could not find the four black squares on the sheet. '
                          'Please place the entire answer sheet inside the camera frame.'),
                'set_version': None, 'answers': {}, 'confidence': {}, 'scores': {}}

    shape_ok, shape_msg, _factor = check_sheet_shape(quad, g)
    if not shape_ok:
        return {'success': False, 'error': shape_msg, 'set_version': None,
                'answers': {}, 'confidence': {}, 'scores': {}}

    if DEBUG_MODE:
        marked = image.copy()
        for idx, (px, py) in enumerate(quad):
            cv2.circle(marked, (int(px), int(py)), 18, (0, 0, 255), 3)
            cv2.putText(marked, ['TL', 'TR', 'BR', 'BL'][idx], (int(px) + 20, int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.polylines(marked, [quad.astype(np.int32)], True, (0, 200, 0), 2)
        _debug_save('03_detected_markers', marked)

    # Everything comes off the fixed grid: the markers are warped onto the
    # corners of a 1536x1345 canvas, then constant bubble centres are read.
    fixed = perspective_correct_fixed(image, quad)
    _debug_save('03c_fixed_canvas', fixed)
    gray = preprocess_image(fixed)

    layout, mark_share = detect_layout(gray)
    if layout is None:
        return {'success': False,
                'error': ('Could not tell which version of the answer sheet this is. '
                          'Retake the photo with the whole sheet flat and evenly lit.'),
                'set_version': None, 'answers': {}, 'confidence': {}, 'scores': {}}

    student_id, id_columns = None, []
    if layout == 'v2':
        answers, confidence, scores = _read_answers(gray, V2_ANSWER_COORDS, V2_ANSWER_RADIUS,
                                                    total_questions, g)
        set_version, set_conf, set_scores = read_set_v2(gray, g)
        student_id, id_columns, _ = read_student_id(gray)
        if DEBUG_MODE:
            _debug_save('06_bubble_rois', _draw_debug_v2(fixed, answers, set_version, id_columns))
    else:
        answers, confidence, scores = _read_answers(gray, FIXED_BUBBLE_COORDS, FIXED_BUBBLE_RADIUS,
                                                    total_questions, g)
        # On these sheets the Set Version column is printed OUTSIDE the marker
        # frame, so it is not on the fixed canvas at all. It keeps the padded
        # warp, which exists precisely to leave room for it.
        warped = perspective_correct(image, quad, g)
        _debug_save('03b_perspective_corrected', warped)
        set_version, set_conf, set_scores = detect_set_version(preprocess_image(warped), g)
        if DEBUG_MODE:
            _debug_save('06_bubble_rois', _draw_debug(warped, answers, g))

    sharpness = measure_sharpness(image)
    return {'success': True, 'error': '', 'layout': layout, 'layout_mark': round(mark_share, 3),
            'set_version': set_version, 'set_confidence': set_conf, 'set_scores': set_scores,
            'student_id': student_id, 'student_id_columns': id_columns,
            'answers': answers, 'confidence': confidence, 'scores': scores,
            'canvas': [FIXED_CANVAS_WIDTH, FIXED_CANVAS_HEIGHT],
            'bubble_count': FIXED_BUBBLE_COUNT,
            'sharpness': round(sharpness, 1),
            'warning': ('The photo is soft; if answers look wrong, retake it steadier.'
                        if sharpness < SHARPNESS_WARN else '')}

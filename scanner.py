"""
OMR + OCR API for the StudyBuddie quiz scanner.

Run: python scanner.py
Listens on:  http://192.168.100.153:5005/scan  (POST multipart/form-data, field: image)

Returns JSON:
    {
      "ok": true,
      "answers":      { "1": "b", "2": "c", ... "30": "" },
      "header": {
          "quiz_number": 1,
          "name":        "Juan Dela Cruz",
          "section":     "BSCS 1A"
      }
    }
or:
    { "ok": false, "error": "..." }

The answer sheet has 30 questions arranged in a 5-row x 6-column grid plus
a header strip with "Quiz N: ...", "Student Name: ___", "Section: ___".
"""

import os
import re
import shutil
import json
import io
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import numpy as np
import cv2

import omr   # marker-based reader: geometry, fill ratios, BLANK/MULTIPLE/AMBIGUOUS

# The OMR service is stateless: it reads a sheet and answers. It used to also
# write every scan into a `scan_results` table, a second copy of data nothing
# ever read. Persistence belongs to the PHP layer, which already stores the
# score in quiz_scores and the per-question read in scan_answers.


try:
    import pytesseract
    from pytesseract import TesseractNotFoundError

    if os.name == 'nt' and not shutil.which('tesseract'):
        for p in [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Local', 'Tesseract-OCR', 'tesseract.exe')
        ]:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    else:
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

app = Flask(__name__)

@app.route("/")
def health():
    return "OK", 200

DEFAULT_QUESTIONS = 40

# A sheet without the four marks cannot be graded by geometry. Refusing is the
# safe default -- see the note in grade_sheet_full().
REQUIRE_MARKERS = omr._env_flag('OMR_REQUIRE_MARKERS', True)
OPTIONS           = ['a', 'b', 'c', 'd']

LAYOUTS = {
    30: (10, 3),
    40: (10, 4),
}


def get_layout(total: int):
    return LAYOUTS.get(total, (5, max(1, (total + 4) // 5)))


def question_number(row: int, col: int, rows: int) -> int:
    return col * rows + row + 1


# --- registration-mark sheet detection ---------------------------------------
# The printed sheet carries four solid black squares whose CENTRES sit exactly on
# the four corners of the bubble grid (see .omr-grid / .reg in final_quiz.php).
# Finding them pins the grid by geometry instead of guessing at the largest
# rectangle in the photo, so how much desk, hand or laptop is in frame stops
# mattering -- which was the single biggest source of misreads.
REG_MIN_AREA_FRAC = 0.00002   # a ~7mm square, photographed small and far away
REG_MAX_AREA_FRAC = 0.03
# Both shape tests run against cv2.minAreaRect, not the axis-aligned bounding
# box, so they are rotation-invariant. That matters: a square tilted 45 degrees
# fills only half of its axis-aligned box, so an axis-aligned test throws real
# marks away as soon as the photo is turned.
REG_SQUARENESS    = 0.75      # short side / long side of the fitted rectangle
# Fraction of the fitted rectangle the blob fills. A square sits near 1.0 at any
# angle; a circle is pi/4 = 0.785. This is what stops the shaded bubbles on an
# older sheet being mistaken for marks, which would offset the grid by half a
# cell and quietly wreck the score.
REG_FILL          = 0.82

# On a registration-marked sheet each question row is laid out as five equal
# columns: [number][A][B][C][D]. So the bubble centres are exact by construction
# rather than measured off a photo.
REG_OPTION_CENTERS_X = (0.30, 0.50, 0.70, 0.90)

# Bubble sampling for the registration-marked sheet, which prints EMPTY circles
# (the A/B/C/D letters sit in the column headings instead). Sampling just the
# circle's interior is what separates shaded from unshaded: an empty bubble is
# plain white inside, a filled one is not.
#
# Measured on rendered sheets photographed under seven conditions. With the
# legacy window (0.12 x 0.60) an unshaded bubble reached 10.9 while the faintest
# pencil mark started at 13.0 -- a 2-level gap against a threshold of 11, so
# paper texture alone was enough to report an answer on a blank row. Sampling
# the interior widens that to roughly 16 levels:
#
#     window        unshaded max   faintest mark   gap
#     0.12 x 0.60          10.9            13.0    2.0   (legacy)
#     0.11 x 0.30          13.3            29.7   16.4   (used here)
#
# Across 280 blank cells this reports 0 answers, versus 3 with the old numbers,
# and reads every shading level from very light pencil to pen at 40/40.
REG_BUBBLE_WIN_W  = 0.11
REG_BUBBLE_WIN_H  = 0.30
REG_MARK_FILL_ABS = 21     # sits mid-gap between an unshaded bubble and a light mark
REG_MARK_FILL_REL = 0.09

# Legacy sheets print the letter inside each bubble, so their calibration is
# different and stays as it was. Each path keeps its own profile:
#   (centres, win width, win height, absolute floor, relative floor, local paper)
REG_BUBBLE_PROFILE = (REG_OPTION_CENTERS_X, REG_BUBBLE_WIN_W, REG_BUBBLE_WIN_H,
                      REG_MARK_FILL_ABS, REG_MARK_FILL_REL, True)

# The marks are printed 6% of the grid's width/height outside each corner, so
# the quad they form is the grid enlarged by 12% in each axis. This is the
# fraction to trim off every side after warping to recover the grid itself.
REG_FRAME_MARGIN = 0.06
REG_GRID_INSET   = REG_FRAME_MARGIN / (1.0 + 2.0 * REG_FRAME_MARGIN)

# A genuine mark is ringed by blank paper. A shaded bubble never is -- it has
# its neighbours, the row box and the question number close by. Comparing the
# ring around a blob against the blob itself is what keeps older, unmarked
# sheets out of this code path.
REG_RING_CONTRAST = 1.9    # ring must be this much brighter than the blob
REG_RING_UNIFORM  = 0.20   # and this uniform (std as a fraction of its mean)

# Opposite sides of the mark quad may differ by at most this much. Perspective
# on a flat sheet stays well inside it; unrelated blobs do not.
REG_SIDE_RATIO = 1.5


def _mark_is_isolated(gray: np.ndarray, cx: float, cy: float, size: float) -> bool:
    """True when the blob at (cx, cy) sits alone on clean paper."""
    r_in = size * 0.80
    r_out = size * 1.75
    x0, y0 = int(cx - r_out), int(cy - r_out)
    x1, y1 = int(cx + r_out), int(cy + r_out)
    h, w = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return False

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cheb = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    ring = patch[(cheb >= r_in) & (cheb <= r_out)]
    core = patch[cheb <= size * 0.35]
    if ring.size < 24 or core.size < 4:
        return False

    ring_mean = float(np.mean(ring))
    core_mean = float(np.mean(core)) + 1.0
    if ring_mean / core_mean < REG_RING_CONTRAST:
        return False
    return float(np.std(ring)) <= REG_RING_UNIFORM * max(ring_mean, 1.0)


def find_registration_marks(image: np.ndarray):
    """Locate the four printed squares.

    Delegates to omr.detect_corner_markers so there is a single implementation.
    Two copies had already drifted apart once; keeping one avoids fixing a bug
    in the marker test and leaving the other copy broken.
    """
    return omr.detect_corner_markers(image)


def find_largest_rect(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image.shape[0] * image.shape[1]
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.05 * img_area:
            return approx.reshape(4, 2)
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_rect(image: np.ndarray, pts: np.ndarray, width: int = 1200, height: int = 800):
    rect = order_points(pts.astype("float32"))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


# --- Bubble-mark detection tuning ------------------------------------------
# Darkness = 255 - mean(blue channel) over a bubble slot's centre. A blank slot
# only has the thin circle outline + printed letter; a shaded slot is far darker.
# --- bubble-mark detection --------------------------------------------------
# The four option bubbles sit at FIXED positions on every printed sheet (the
# template is identical for all 12 quizzes and sets A/B/C). Sampling a window
# centred on each true bubble centre -- instead of naively splitting the cell
# into four equal slots -- is what makes detection consistent from photo to
# photo. The old even-split assumed the number label was 12% wide; it is wider,
# so the whole grid of slots sat left of the real bubbles and first-column (A)
# answers landed on the A/B seam and were missed. These centres were measured
# from rectified sheets (A 0.33, B 0.52, C 0.71, D 0.90 of the cell width).
OPTION_CENTERS_X = (0.332, 0.518, 0.707, 0.902)  # bubble centres, fraction of cell width
BUBBLE_WIN_W     = 0.12   # sampling window width,  fraction of cell width
BUBBLE_WIN_H     = 0.60   # sampling window height, fraction of cell height

# Each bubble is then judged by its MEAN darkness relative to the lightest
# bubble (the row's blank paper). A fully shaded circle darkens the whole window
# -> HIGH fill -> accepted; a thin line or partial mark barely moves the average
# -> LOW fill -> rejected. The printed A/B/C/D letters darken every bubble about
# equally, so they cancel out of the comparison.
MARK_FILL_ABS    = 11     # winner must be >= this many 0-255 levels darker than the lightest bubble
MARK_FILL_REL    = 0.06   # ...and >= 6% darker relative to that paper level
MARK_AMBIG_RATIO = 0.68   # a 2nd bubble nearly as dark => ambiguous (double mark / near-blank row)

# Set OMR_DEBUG=1 in the environment to print per-row fill stats.
OMR_DEBUG = os.getenv('OMR_DEBUG', '').strip() not in ('', '0', 'false', 'False')


def detect_bubbles_in_cell(cell: np.ndarray, profile=None) -> str:
    """
    Answer-sheet row layout:  [ Q-num | Ⓐ | Ⓑ | Ⓒ | Ⓓ ]

    Sample a fixed-size window centred on each of the four true bubble positions
    and pick the bubble that is meaningfully darker (on average) than the
    lightest one. A fully shaded circle fills its window and reads dark; a thin
    line or partial mark leaves most of the window light, so it is rejected.
    """
    if cell.size == 0:
        return ''
    h, w = cell.shape[:2]
    if h < 8 or w < 20:
        return ''

    # Blue channel keeps the blue box borders bright (so they are ignored) while
    # pencil marks stay dark -- the printed grid does not interfere.
    chan = cell[:, :, 0] if len(cell.shape) == 3 else cell

    centers, p_win_w, p_win_h, p_fill_abs, p_fill_rel, p_local_paper = (
        profile or (OPTION_CENTERS_X, BUBBLE_WIN_W, BUBBLE_WIN_H,
                    MARK_FILL_ABS, MARK_FILL_REL, False))

    win_w = max(4, int(p_win_w * w))
    win_h = max(4, int(p_win_h * h))
    y0 = (h - win_h) // 2

    # Clean paper directly above and below each bubble, inside the same row.
    band_h = max(2, int(0.16 * h))
    top_a, top_b = int(0.02 * h), int(0.02 * h) + band_h
    bot_a, bot_b = h - int(0.02 * h) - band_h, h - int(0.02 * h)

    means, locals_ = [], []
    for cx in centers:
        x0 = int(cx * w) - win_w // 2
        x0 = max(0, min(w - win_w, x0))
        win = chan[y0:y0 + win_h, x0:x0 + win_w]
        means.append(float(np.mean(win)) if win.size else 255.0)
        if p_local_paper:
            strips = [chan[top_a:top_b, x0:x0 + win_w], chan[bot_a:bot_b, x0:x0 + win_w]]
            vals = [s for s in strips if s.size]
            locals_.append(float(np.mean(np.concatenate([v.ravel() for v in vals])))
                           if vals else 255.0)

    if p_local_paper:
        # Compare every bubble against the paper beside IT. Taking the brightest
        # bubble as the reference for the whole row turns an illumination
        # gradient into a fake answer: on a real blank sheet photographed at an
        # angle the row fell 21 levels from A to D, which alone cleared the
        # threshold and reported D. Measuring at the same x cancels that.
        paper = max(locals_) if locals_ else 255.0
        if paper <= 1:
            return ''
        fill = [max(0.0, lp - m) for lp, m in zip(locals_, means)]
    else:
        paper = max(means)                   # lightest bubble = local blank paper
        if paper <= 1:
            return ''
        fill = [paper - m for m in means]    # how much darker than paper (>= 0)
    chosen = int(np.argmax(fill))
    top = fill[chosen]
    srt = sorted(fill, reverse=True)
    runner_up = srt[1] if len(srt) > 1 else 0.0

    valid = (top >= p_fill_abs
             and (top / paper) >= p_fill_rel
             and runner_up <= MARK_AMBIG_RATIO * top)

    if OMR_DEBUG:
        print(f"[omr] means={[round(m, 1) for m in means]} fill={[round(f, 1) for f in fill]} "
              f"paper={paper:.1f} top={top:.1f} rel={top / paper:.3f} run={runner_up:.1f} "
              f"-> {OPTIONS[chosen] if valid else 'BLANK'}", flush=True)

    # Not clearly darker than the surrounding paper -> blank / thin line / partial.
    if top < p_fill_abs or (top / paper) < p_fill_rel:
        return ''
    # A second bubble almost as dark -> ambiguous (double mark / near-blank row).
    if runner_up > MARK_AMBIG_RATIO * top:
        return ''

    return OPTIONS[chosen]


# --- checkbox-mark detection (experimental sample sheets) -------------------
# The sample sheets use an empty SQUARE per option instead of a bubble, with the
# A/B/C/D letter printed OUTSIDE the box, so a checked square is the only thing
# with ink inside it. That makes a clean AREA-COVERAGE test reliable here: the
# checked box has a clear patch of ink, an empty one has almost none. Exactly one
# clearly-marked box is accepted; nothing, or two comparable marks, is rejected.
CHECK_INK_DROP  = 0.30   # a pixel is "ink" if it is >= 30% darker than paper white
CHECK_COVER_MIN = 0.06   # >= 6% of the box interior inked => a mark is present
CHECK_LEAD      = 2.2    # winning box coverage must be >= this x the runner-up


def detect_checkbox_in_cell(cell: np.ndarray) -> str:
    """Detect a hand-drawn check inside one of the four square checkboxes."""
    if cell.size == 0:
        return ''
    h, w = cell.shape[:2]
    if h < 8 or w < 20:
        return ''
    chan = cell[:, :, 0] if len(cell.shape) == 3 else cell

    white = float(np.percentile(chan, 85))
    if white < 1.0:
        white = 1.0
    ink_level = white * (1.0 - CHECK_INK_DROP)

    win_w = max(4, int(0.11 * w))
    win_h = max(4, int(0.55 * h))
    y0 = (h - win_h) // 2
    cover = []
    for cx in OPTION_CENTERS_X:
        x0 = int(cx * w) - win_w // 2
        x0 = max(0, min(w - win_w, x0))
        box = chan[y0:y0 + win_h, x0:x0 + win_w]
        cover.append(float(np.count_nonzero(box < ink_level)) / box.size if box.size else 0.0)

    chosen = int(np.argmax(cover))
    top = cover[chosen]
    srt = sorted(cover, reverse=True)
    runner_up = srt[1] if len(srt) > 1 else 0.0

    valid = (top >= CHECK_COVER_MIN and runner_up * CHECK_LEAD <= top)
    if OMR_DEBUG:
        print(f"[chk] cover={[round(c, 3) for c in cover]} top={top:.3f} run={runner_up:.3f} "
              f"-> {OPTIONS[chosen] if valid else 'BLANK'}", flush=True)

    if top < CHECK_COVER_MIN:            # no mark inside any box
        return ''
    if runner_up * CHECK_LEAD > top:     # two comparable marks -> ambiguous
        return ''
    return OPTIONS[chosen]


def parse_header_text(text: str) -> dict:
    out = {'quiz_number': None, 'last_name': '', 'first_name': '', 'middle_name': '', 'year_section': ''}
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    full = '\n'.join(lines)

    def clean_text(raw_str):
        c = re.sub(r'[_=]+', ' ', raw_str)
        c = re.sub(r'\s+', ' ', c).strip(" .-,'")
        return c if any(ch.isalpha() or ch.isdigit() for ch in c) else ''

    m = re.search(r'Quiz\s*[:\-]?\s*(\d{1,2})', full, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Quiz\s*#?\s*(\d{1,2})', full, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Chapter\s*(\d{1,2})', full, flags=re.IGNORECASE)
        
    if m:
        n = int(m.group(1))
        if 1 <= n <= 12:
            out['quiz_number'] = n

    m = re.search(r'Last\s*Name\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
    if m:
        raw = re.split(r'(?:First\s*Name|Middle\s*Name|Year|Quiz|Date)\s*[:\-]?', m.group(1), flags=re.IGNORECASE)[0]
        out['last_name'] = clean_text(raw)[:100]

    m = re.search(r'First\s*Name\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
    if m:
        raw = re.split(r'(?:Middle\s*Name|Last\s*Name|Year|Quiz|Date)\s*[:\-]?', m.group(1), flags=re.IGNORECASE)[0]
        out['first_name'] = clean_text(raw)[:100]

    m = re.search(r'Middle\s*Name\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
    if m:
        raw = re.split(r'(?:First\s*Name|Last\s*Name|Year|Quiz|Date)\s*[:\-]?', m.group(1), flags=re.IGNORECASE)[0]
        out['middle_name'] = clean_text(raw)[:100]

    m = re.search(r'Year\s*(?:&|and|%)\s*Section\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Year\s*/\s*Section\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'Section\s*[:\-]?\s*([^\n]+)', full, flags=re.IGNORECASE)
        
    if m:
        raw = re.split(r'(?:Date|Last\s*Name|First\s*Name|Quiz)\s*[:\-]?', m.group(1), flags=re.IGNORECASE)[0]
        out['year_section'] = clean_text(raw)[:100]

    return out


def ocr_header(header_img: np.ndarray) -> dict:
    """OCR the top header strip; pull out Quiz, Last Name, First Name, Middle Name, Year & Section."""
    if not OCR_AVAILABLE or header_img.size == 0:
        return {'quiz_number': None, 'last_name': '', 'first_name': '', 'middle_name': '', 'year_section': ''}

    gray = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY) if len(header_img.shape) == 3 else header_img
    h, w = gray.shape
    scale = max(1.0, 1500.0 / max(h, w))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    proc = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    try:
        text = pytesseract.image_to_string(proc, config='--psm 6')
        return parse_header_text(text)
    except Exception:
        return {'quiz_number': None, 'last_name': '', 'first_name': '', 'middle_name': '', 'year_section': ''}


def qr_header(img: np.ndarray) -> dict:
    """Detects and decodes QR code in the image."""
    try:
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if data:
            try:
                js = json.loads(data)
                return {
                    'quiz_number': int(js.get('quiz_number') or js.get('quiz') or 0) or None,
                    'last_name': str(js.get('last_name') or '')[:100],
                    'first_name': str(js.get('first_name') or '')[:100],
                    'middle_name': str(js.get('middle_name') or '')[:100],
                    'year_section': str(js.get('year_section') or js.get('section') or '')[:100],
                }
            except Exception:
                pass
            return parse_header_text(data)
    except Exception:
        pass
    return None


def grade_sheet_full(image_bytes: bytes, total_questions: int = DEFAULT_QUESTIONS,
                     mode: str = 'bubble'):
    """Read a sheet and return everything known about the read.

    Marked sheets go through omr.py, which locates the grid from the four
    printed squares and reports BLANK / MULTIPLE / AMBIGUOUS rather than
    guessing. Sheets printed before the squares existed fall back to the
    original path, so nothing already in circulation stops working.
    """
    empty = {'ok': False, 'answers': {}, 'status': {}, 'confidence': {},
             'set_version': None, 'student_id': None, 'layout': None,
             'header': None, 'error': '', 'engine': 'none'}

    try:
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        empty['error'] = f'Could not decode image: {e}'
        return empty

    # Work at a bounded size for speed; the bubbles stay large enough to read.
    max_dim = 2200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Refuse an unusable photo before either engine runs, so a dark, tiny or
    # blown-out frame comes back as advice rather than as invented answers.
    usable, why = omr.validate_scan(img)
    if not usable:
        empty['error'] = why
        return empty

    header_info = qr_header(img)
    if not header_info or not any([header_info.get('last_name'), header_info.get('first_name')]):
        header_info = ocr_header(img[:int(img.shape[0] * 0.4), :])

    quad = omr.detect_corner_markers(img) if mode == 'bubble' else None

    if quad is not None:
        result = omr.scan_answer_sheet(img, total_questions=total_questions, quad=quad)
        if not result['success']:
            empty['error'] = result['error']
            empty['header'] = header_info
            empty['engine'] = 'markers'
            return empty

        # Keep the wire format the rest of the system already expects: a plain
        # letter, or '' for anything not confidently a single answer. The
        # richer verdict travels alongside in `status`.
        answers, status = {}, {}
        for qn, verdict in result['answers'].items():
            status[qn] = verdict
            answers[qn] = verdict if verdict in OPTIONS else ''
        return {'ok': True, 'answers': answers, 'status': status,
                'confidence': result['confidence'], 'scores': result.get('scores', {}),
                'set_version': result['set_version'], 'header': header_info,
                'student_id': result.get('student_id'), 'layout': result.get('layout'),
                'canvas': result.get('canvas'), 'bubble_count': result.get('bubble_count'),
                'error': '', 'engine': 'markers'}

    # ---- no registration squares found ----
    # The legacy reader has nothing to align to: it guesses the grid from the
    # largest rectangle, and when even that fails it treats the whole frame as
    # the answer grid. On a sheet that really does carry the four marks, landing
    # here means the photo was unusable -- and a plausible-looking wrong score is
    # far worse than a refusal, because it gets saved. Refuse by default; set
    # OMR_REQUIRE_MARKERS=0 to allow the old best-effort behaviour.
    if mode == 'bubble' and REQUIRE_MARKERS:
        empty['error'] = ('Could not find the four black squares on the answer sheet. '
                          'Put the whole sheet inside the frame, hold the phone flat and '
                          'steady, and keep glare off the corners.')
        empty['header'] = header_info
        empty['engine'] = 'markers'
        return empty

    answers, header_legacy, err = grade_sheet(image_bytes, total_questions, mode)
    if err:
        empty['error'] = err
        return empty
    return {'ok': True, 'answers': answers or {},
            'status': {q: (v if v else 'BLANK') for q, v in (answers or {}).items()},
            'confidence': {}, 'scores': {},
            'set_version': None, 'student_id': None, 'layout': None,
            'header': header_legacy or header_info,
            'error': '', 'engine': 'legacy'}


def grade_sheet(image_bytes: bytes, total_questions: int = DEFAULT_QUESTIONS, mode: str = 'bubble'):
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, f'Could not decode image: {e}'

    if img is None:
        return None, None, 'Could not decode image. Use JPG, PNG, or WebP.'

    max_dim = 2200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    rows, cols = get_layout(total_questions)
    sheet_h = max(800, rows * 80)
    sheet_w = max(1200, 200 * cols)

    # Preferred path: the sheet's four registration marks bound the bubble grid
    # exactly, so warping them gives the grid with no header strip to guess at.
    reg_quad = find_registration_marks(img)
    bubble_profile = None

    if reg_quad is not None:
        # The quad is the grid plus its printed margin, so warp generously and
        # then trim that margin off to land exactly on the bubble grid.
        pad = REG_GRID_INSET
        frame_w = int(sheet_w / (1.0 - 2.0 * pad))
        frame_h = int(sheet_h / (1.0 - 2.0 * pad))
        frame = warp_to_rect(img, reg_quad, width=frame_w, height=frame_h)
        x0, y0 = int(pad * frame_w), int(pad * frame_h)
        grid = frame[y0:frame_h - y0, x0:frame_w - x0]
        bubble_profile = REG_BUBBLE_PROFILE
    else:
        # Legacy sheets with no marks: fall back to the largest rectangle, and
        # to the whole frame if even that fails.
        rect = find_largest_rect(img)
        if rect is None:
            h, w = img.shape[:2]
            rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        sheet = warp_to_rect(img, rect, width=sheet_w, height=sheet_h)
        header_frac = 0.12 if cols <= 2 else 0.14
        header_h = int(sheet.shape[0] * header_frac)
        grid = sheet[header_h:, :]

    header_info = qr_header(img)

    if not header_info or not any([header_info.get('last_name'), header_info.get('first_name')]):
        orig_h = img.shape[0]
        header_info = ocr_header(img[:int(orig_h * 0.4), :])

    gh, gw = grid.shape[:2]
    cell_w = gw // cols
    cell_h = gh // rows

    answers = {}
    detector = detect_checkbox_in_cell if mode == 'checkbox' else detect_bubbles_in_cell

    debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    if os.path.exists(debug_dir):
        cv2.imwrite(os.path.join(debug_dir, 'debug_grid.jpg'), grid)
        
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = (c + 1) * cell_w
            y1 = (r + 1) * cell_h
            pad = max(2, cell_w // 50)
            cell = grid[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
            qn = question_number(r, c, rows)
            
            if qn == 1 and os.path.exists(debug_dir):
                cv2.imwrite(os.path.join(debug_dir, 'debug_cell_q1.jpg'), cell)
                
            if qn <= total_questions:
                answers[qn] = (detector(cell, bubble_profile)
                               if detector is detect_bubbles_in_cell else detector(cell))

    return answers, header_info, None


@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'ok': False, 'error': 'No "image" file in request.'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({'ok': False, 'error': 'Empty image upload.'}), 400

    try:
        total_q = int(request.form.get('total_questions', DEFAULT_QUESTIONS))
    except (TypeError, ValueError):
        total_q = DEFAULT_QUESTIONS
    if total_q not in (30, 40):
        total_q = DEFAULT_QUESTIONS

    mode = (request.form.get('mode', 'bubble') or 'bubble').strip().lower()
    if mode not in ('bubble', 'checkbox'):
        mode = 'bubble'

    try:
        result = grade_sheet_full(image_bytes, total_questions=total_q, mode=mode)
        if not result['ok']:
            return jsonify({'ok': False, 'error': result['error']}), 200

        answers = result['answers']
        resolved_header = result['header'] or {'quiz_number': None, 'last_name': '', 'first_name': '', 'middle_name': '', 'year_section': ''}

        # `answers` keeps its original meaning -- a letter, or '' for anything
        # that is not confidently one answer -- so existing callers are
        # unaffected. Everything else is additive.
        return jsonify({
            'ok': True,
            'answers': {str(k): v for k, v in answers.items()},
            'header': resolved_header,
            'ocr_available': OCR_AVAILABLE,
            'total_questions': total_q,
            'status': {str(k): v for k, v in result.get('status', {}).items()},
            'confidence': {str(k): v for k, v in result.get('confidence', {}).items()},
            'set_version': result.get('set_version'),
            # users.id off the sheet's Student ID bubbles; None on older sheets
            # or when the columns do not read as one clean number.
            'student_id': result.get('student_id'),
            'layout': result.get('layout'),
            'engine': result.get('engine'),
            # Fixed-grid geometry the answers were read from: [w, h] of the
            # rectified canvas and how many bubble coordinates were sampled.
            'canvas': result.get('canvas'),
            'bubble_count': result.get('bubble_count'),
        }), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Processing failed: {e}'}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'service': 'studybuddie-omr', 'ocr_available': OCR_AVAILABLE})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print("--- StudyBuddie OMR Scanner Service ---")
    print(f"Service starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

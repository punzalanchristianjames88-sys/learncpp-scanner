# Quiz Paper Scanner

Mobile-friendly OMR (Optical Mark Recognition) scanner for the StudyBuddie quiz papers. The frontend (PHP) lets a student or admin take a photo of a printed answer sheet on their phone. The PHP backend forwards the image to a small Python service that detects which bubbles are shaded, scores the paper against the answer key, saves the result to the `quiz_scores` table, and unlocks the next chapter for the logged-in student.

## Flow

```
Mobile browser / scanner app
    -> scanner/index.php       (upload page)
    -> scanner/process.php     (PHP: who is scanning, forwards image to Python)
        -> http://127.0.0.1:5005/scan   (Python Flask + OpenCV)
        <- JSON of detected answers, Set and Student ID
    -> score against the uploaded answer sheet for that quiz + set,
       else the chapter's static key, else quiz_answer_keys
    -> compare the sheet's Student ID with the student's account (see below)
    -> save row in quiz_scores (status = "reviewed", source = "scanner")
       and the percentage in users.quizN -- unless the score is held
    -> record the scan in scan_submissions
    -> scanner/result.php (web) or the app's result screen
    -> Submit: result.php / finalize.php -> scanner_submit() re-checks it
```

## Who is scanning, and the Student ID check

The website uses its PHP session. The scanner app has none, so `student_api_login.php`
gives it a token at sign-in (only the token's SHA-256 is stored, in `scanner_api_tokens`,
valid 30 days), and the app sends it with every scan and submit (`Authorization: Bearer`
or an `api_token` field). `scanner_auth.php` resolves the caller from either; a posted
`student_id` or `role` is never trusted. Without a valid session or token, `process.php`
and `finalize.php` answer `code: "auth_required"`, and the app signs out.

- **A student always scans as themselves.** The Student ID read off the sheet is compared
  with their `users.id` (`scanner_id_check()`): `match`, `mismatch`, or `unverified` when the
  sheet has no readable ID (older layout, or printed blank and photocopied) -- never a
  mismatch. An instructor's scans are `staff` and are not compared.
- **Scanning never reports anyone.** A mismatched scan is recorded in `scan_submissions`
  but its score is held: nothing is written to `quiz_scores` / `users.quizN`, no chapter
  unlocks, and the student sees a neutral note.
- **Submit decides, server-side** (`scanner_submit()` in `scan_submissions.php`, from the
  stored record, not from what the client sends). A match or unverified scan finalizes as
  before. A mismatch opens a `scan_incidents` row with status `PENDING_REVIEW` -- the
  submitter, the ID on the sheet, the account it belongs to if any, quiz, set, held score,
  reason, time -- and one `instructor_notifications` row for the submitter's instructor.
  The quiz is not locked and the key is not revealed. Submitting the same scan again, or
  another scan of the same sheet while the case is open, joins that incident
  (`attempts`) instead of notifying again.
- **The instructor decides** in *Scan Reviews* (`admin/scan_reviews.php`; the sidebar
  badge counts pending cases). *Accept* counts the held score and finalizes the quiz;
  *Reject* discards it and leaves the quiz open. Neither touches the other student's
  account, and nothing else is penalised automatically.
- An instructor may only scan or submit for students whose `instructor_id` is theirs.

## Setup (Windows / XAMPP)

### 1. Install Python 3.10+

Download from https://www.python.org/downloads/. During install, tick "Add python.exe to PATH".

### 2. Install Python dependencies

Open PowerShell in this folder:

```powershell
cd c:\xampp\htdocs\studybuddie\scanner\python
python -m pip install -r requirements.txt
```

### 3. Start the OMR service

Use the helper script from the Python folder:

```powershell
cd c:\laragon\www\LearnC++\modules\scanner\python
auto_start_service.bat
```

Or run the service manually:

```powershell
python scanner.py
```

You should see:
```
 * Running on http://0.0.0.0:5005
```

Leave this terminal open while you scan papers.

### 4. (Optional) Open phone access

If you want to scan from your phone instead of the PC's browser:

1. Make sure phone and PC are on the same Wi-Fi.
2. Find the PC's local IP (e.g. `ipconfig` -> IPv4 Address, often `192.168.x.x`).
3. Allow port 80 (Apache) and 5005 (Python) in Windows Firewall.
4. On the phone, open `http://<PC-IP>/studybuddie/scanner/`.

PHP talks to Python on `127.0.0.1:5005` from the server itself, so the phone only needs to reach Apache.

## Usage

1. Open `scanner/index.php` in a phone browser.
2. The live camera opens automatically; align the answer sheet in the guide and tap the shutter.
3. Capture the **ANSWER SHEET (Q1-Q30)** box (the boxed grid in the printed quiz).
4. Fill in **Last Name**, **First Name**, **Year & Section**, choose the **Quiz number**.
5. Tap **Scan & Submit**.
6. The result page shows the score and which questions were correct / wrong / blank.
7. The score is automatically saved and visible in the admin dashboard.

## Answer sheets (`answer_sheets/`)

Instead of typing a key in, the instructor can shade a printed sheet with the correct
answers and upload the photo. The scanner reads that photo the same way it reads a
student's paper, and every scan of that quiz is compared against it.

1. Log in as admin -> **Quiz Answer Key** -> pick the chapter and the set (A, B or C).
2. Under **Scanner Answer Sheet**, upload the photo of the shaded sheet.
3. The page shows the letters that were read. Red chips are questions where no single
   bubble was detected — those count as wrong for everyone, so re-upload a clearer
   photo if any show up.
4. **Remove this answer sheet** deletes it and grading falls back to the built-in key.

The images live in `answer_sheets/`, one per quiz and set (`quiz3_set_b.jpg`). The read
itself is cached in the `scanner_answer_sheets` table, so grading a student never
re-scans the key. A photo that fails to read is discarded and the sheet already on file
is left alone, so a bad upload can't wipe a working key.

Lookup order used by `scanner_load_answer_key()` in `quiz_key.php`:

1. `answer_sheets/` — the uploaded sheet for this quiz + set
2. `modules/lessons/chapter{N}/final_quiz_data_set_{a,b,c}.php`
3. `modules/lessons/chapter{N}/final_quiz_data.php`
4. the `quiz_answer_keys` table

Both the website and the Android app go through `process.php`, so they grade against the
same uploaded sheet.

## How the OMR works

Current sheets print four black registration squares, one per corner. Those go through
`python/omr.py`:

```
photo
  -> detect the 4 corner squares          (adaptive threshold; OPEN retry, see below)
  -> perspective correction               (the 4 markers -> the canvas corners)
  -> 1536 x 1345 rectified canvas
  -> which layout                         (the layout bar, see below)
  -> 160 fixed bubble coordinates         (40 questions x 4 options, constants)
  -> OMR analysis                         (per-bubble local paper reference)
  -> answers 1-40  + BLANK / MULTIPLE / AMBIGUOUS
  -> Set + Student ID                     (current sheets)
  -> JSON result
```

Because the markers are warped onto the canvas corners every time, a bubble lands on the
same pixel on every sheet. The centres are therefore constants, built once from the printed
layout rather than re-derived per photo. There are two layouts:

- **Current sheets** (`SHEET_V2`, built by `_build_v2_coords()`): the Set and Student ID
  bubbles sit *inside* the marker frame, in a panel left of the answers. Every block is
  placed in percentages of the frame, the same percentages `.omr-frame` uses in
  `final_quiz.php`. Each answer row is `[number 0.6][A][B][C][D]`; Q1's A bubble is at
  (455.2, 196.22) with radius 20.5.
- **Older sheets** (`SHEET`, `_build_fixed_bubble_coords()`): the grid fills the frame with
  the markers 6% outside it, so the grid insets to (82.3, 72.1) and each cell is
  342.86 x 120.09; Q1's A bubble is at (185.14, 132.10) with radius 22.1. Their Set Version
  column is printed outside the frame, so it is not on the fixed canvas and keeps its own
  padded warp; they have no Student ID.

`detect_layout()` tells them apart by the **layout bar**, a solid bar current sheets print
at the bottom centre of the frame, on a spot that is blank paper on older sheets. Ink there
means current, paper means older, and anything in between (glare, a scribble) refuses the
scan, because reading one layout's bubbles off the other would grade every answer against
the wrong spot.

**Student ID.** Five columns of 0-9 bubbles carry `users.id`, first digit in the first
column (125 shades 1, 2, 5 and leaves columns 4-5 blank); the print page shades them.
`read_student_id()` returns the number only when every column reads cleanly -- one mark per
column, nothing after the first blank column, no leading zero -- and `None` otherwise. When
an instructor picks no student ("Use the Student ID on the sheet"), `process.php` identifies
the student by that id, if it exists in `users`; a picked student always wins. On a
student's own scan the id is compared with their account instead (see "Who is scanning"
above). The id read off the sheet is reported as `detected_student_id` either way. Grading
still uses the account's set (`quiz_set_for_user()`), not the shaded Set bubble.

Everything the reader uses sits inside the frame on purpose. The detector takes the
candidate furthest toward each corner as that corner, and in a blurry photo a shaded bubble
can pass for a square -- one printed outside the frame, toward a corner, can replace that
square. Older sheets have exactly that problem with a shaded Set bubble near the top-left
square.

Thresholding is **not** global. `calculate_fill_ratio()` takes the paper level from a ring
just outside each bubble and marks ink below `paper x (1 - 0.18)`, so shadow across the
page cannot fake an answer. A circular inner mask at 0.62r keeps the printed outline out of
the measurement. The fixed constants (`MARK_THRESHOLD` etc.) act on the resulting fill
*fraction*, which is already illumination-independent.

**Marker rescue.** Printing that runs close to a marker can weld itself onto it — on older
sheets the rotated instruction text sits a few pixels from the two top squares (current
sheets print the instructions below the frame instead), the threshold joins
them, and the merged blob is no longer square. When that leaves fewer than four markers,
detection retries on an `MORPH_OPEN`ed binary, which severs thin strokes but leaves a solid
square intact. The retry is deliberately hard to trigger, because the same OPEN also frees
a shaded bubble from its row box and a blurred bubble is not distinguishable from a blurred
square by shape or darkness. It is only trusted when all three hold: the clean pass already
found at least 2 markers, the retry finds exactly 4, and they form a near-rectangle
(`MARKER_RESCUE_SIDE_RATIO`).

### Which sheet to print

Print the answer sheet from the **quiz page** (`modules/lessons/final_quiz.php`), opened by an
instructor through View Chapters -- students are redirected away from it. Its
`.omr-frame` is what every current-sheet coordinate in `omr.py` is derived from: marks whose
centres sit exactly on the frame corners, aspect 800:710 (the shape both scanner apps crop
to), the Set, Student ID and answer blocks at fixed percentages of the frame, and —
importantly — **empty circles**, with the A/B/C/D letters in the column headings. Letters
printed *inside* the bubbles put ink in all four options and destroy the contrast the reader
depends on.

The bubble grid rendered on `admin/quiz_answer_key.php` is an on-screen **preview**, not a
printable sheet: its four anchors bound the whole card, title strip and padding included,
so the grid does not sit at the 6% inset the reader assumes. Photographing it puts every
ROI about one row too high. The shape check below now catches this instead of quietly
scoring it wrong.

### Refusing a scan rather than guessing

Three gates stop an unreliable read before it can reach the database:

| Gate | Rejects |
|---|---|
| `validate_scan()` | too small, too dark, blown out |
| four markers | sheet not fully in frame, corners clipped or lost to glare |
| `check_sheet_shape()` | markers found but the frame is the wrong aspect — a different sheet, or false corners |

Measured across 115 real photos where markers were found: the 24 inside the shape band read
30.5 of 40 on average; the 91 outside never got past 18. Set `OMR_REQUIRE_MARKERS=0` to fall
back to the old best-effort legacy reader instead of refusing.

### Tuning without editing code

Every threshold reads an environment variable, defaulting to the calibrated value:

`OMR_MARK_THRESHOLD` 0.30 · `OMR_AMBIGUOUS_MARGIN` 0.12 · `OMR_MULTIPLE_THRESHOLD` 0.30 ·
`OMR_INK_DROP` 0.18 · `OMR_ASPECT_TOLERANCE` 1.75 · `OMR_MIN_RESOLUTION` 300 ·
`OMR_MIN_MEAN_LEVEL` 35 · `OMR_MAX_MEAN_LEVEL` 245 · `OMR_SHARPNESS_WARN` 25

### Diagnosing one image

`python/omr_debug.py` runs a single image through the real pipeline and prints what every
bubble measured, so a wrong answer can be traced to the stage that caused it:

```powershell
cd modules\scanner\python
python omr_debug.py "..\quiz 1 set a.png"
```

```
IMAGE     quiz 1 set a.png  836x732
QUALITY   ok
MARKERS   found  TL(42,33)  TR(800,33)  BR(800,706)  BL(42,706)
SHEET     1.00x the printed aspect  (accepted band 0.57-1.75)
CANVAS    1536x1345  160 bubble coordinates, radius 22.1

Question 1:
A = 100%
B = 0%
C = 0%
D = 0%
Answer = A
Confidence = 1.0
```

It also writes four stage images — original, detected markers, perspective-corrected canvas,
and the 160 sampled regions drawn over it — to `uploads/debug/` or `--out`.

### Legacy sheets (no registration squares)

Sheets printed before the squares existed fall back to `grade_sheet()` in `scanner.py`:
largest 4-corner contour -> flat rectangle -> split into cells -> compare each bubble's mean
darkness against the lightest bubble in its row. This path is less robust and is kept only
so older printouts keep working.

## Configuration

- **Python URL** - override per environment by setting the env var `OMR_API_URL`, e.g. `OMR_API_URL=http://10.0.0.5:5005/scan`. Default is `http://127.0.0.1:5005/scan`.
  - If `OMR_API_URL` does not include `/scan`, the backend will append it automatically.
- **Health endpoint** - optionally set `OMR_API_HEALTH_URL` when your OMR service exposes a custom health route.
- **Answer key** - chapter 1 ships with the static defaults from `admin/quiz_answer_key.php`. For other chapters, upload an answer sheet (above) or set the answers in the admin panel before scanning.
- **No scan history** - a student's photographed sheet is never stored. It is written to the system temp directory only for as long as cURL needs to stream it to the OMR service, then deleted on shutdown however the request ends. What is kept is the score in `quiz_scores` and the per-question read in `scan_answers`. `scanner/uploads/` now only holds `debug/` output when `OMR_DEBUG_IMAGES=1`.
- **Answer sheets** - the instructor's uploaded key sheets live in `scanner/answer_sheets/` and *are* kept, since they are the answer key. Add that folder to backups.

## Troubleshooting

- **"Could not reach the OMR service"** - the Python script isn't running. Open a terminal and run `python scanner.py`.
- **All answers come back blank** - the camera shot probably did not capture the full ANSWER SHEET box, or it was very tilted. Retake with the paper flat and well lit.
- **A few Qs detected wrong** - the simulator only catches obvious shading. Lightly drawn marks may not register. Tell students to fill bubbles fully with a dark pen or pencil.
- **"No answer key has been set for Quiz N yet"** - log in as admin, open Quiz Answer Key, choose the chapter, upload the answer sheet or save the answers, then re-scan.
- **Everyone suddenly scores badly on one quiz** - check Quiz Answer Key for that chapter and set. If an answer sheet is uploaded, look at the red chips: questions the OMR could not read off the key are counted wrong for every student. Re-upload a clearer photo or remove the sheet.

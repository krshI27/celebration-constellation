# Implementation Plan: Compact Auto-Pipeline UI

## Objectives

- Single, top-of-page image viewer with in-main viewer layer toggles; sidebar reserved for processing parameters only.
- Automatic end-to-end pipeline on upload: show uploaded image immediately, then stream results (dark greyscale, synthwave greyscale, circles, matches, constellation overlays) as they complete.
- Layer availability follows data readiness; background choices expand as outputs arrive.
- Mini starmap includes input-image footprint rectangle for the selected match.
- Reduce non-essential text; keep interface compact.
- Parameter changes happen via a grouped form with one “Run” button; do not auto-rerun on every tweak.
- Initial background defaults to pure black until derived backgrounds are ready.
- For iconified buttons, use Font Awesome (no emoji) and avoid mixing icon + text on the same control.

## Workstream Overview (ordered by dependency/priority)

1. **UI Layout & State Skeleton (P0)**
   - Collapse to single top viewer section; move layer toggles (background, circles, constellations, star colors) into main area.
   - Keep sidebar for processing knobs only (detection/matching sizing and sensitivity parameters).
   - Trim explanatory text; keep concise status/progress labels.
   - If a control uses an icon, prefer Font Awesome and do not pair with text on the same button.
2. **Upload Auto-Start & Stage Flags (P0)**
   - On file upload, immediately show RGB image and kick off pipeline.
   - Add session-state flags for stage readiness: `bg_ready_dark`, `bg_ready_synth`, `circles_ready`, `matches_ready`.
   - Surface incremental status in main viewer (e.g., small status pill or text row).
   - Initial render uses pure black background until generated backgrounds are available.
3. **Background Generation Staging (P0)**
   - Compute dark greyscale and synthwave greyscale as first async step; expose as soon as ready.
   - Initial selectable backgrounds: black and linear synthwave (fallback); expand to greyscale/synthwave-from-image when ready; keep original RGB if desired.
4. **Auto Circle Detection (P0)**
   - Trigger detection immediately after background conversions complete; preserve current parameterization (object size auto, sensitivity slider) from sidebar.
   - Update stage flag `circles_ready`; allow circle layer toggle once ready.
5. **Auto Constellation Matching (P1)**
   - Start matching right after circles are available; remove manual "Find Matching Constellations" CTA.
   - Keep progress feedback in main area; update `matches_ready` when done.
   - Prev/Next match navigation remains; ensure viewer reflects selected match.
   - Respect parameter form submissions (see below) to rerun with updated settings instead of live-recomputing on every change.
6. **Parameter Form & Run Control (P1)**
   - Group processing parameters (detection sensitivity, object size preset/auto radius, sky regions/window, brightness/size if treated as processing) into a form.
   - Provide a single "Run" button to apply changes and rerun detection/matching; do not trigger on every slider/selector change.
   - Keep current session outputs visible while edits are in progress; rerun overwrites stage flags/results when submitted.
6. **Layer Control Logic (P1)**
   - Disable/grey layer selectors until prerequisites are ready (backgrounds → circles → constellations).
   - Background selector ordering: Black/Synthwave (default), Dark Greyscale (when ready), Synthwave-from-image, Original RGB (optional), Annotated detection.
   - Star color mode & constellation line toggles active only after matches ready.
7. **Mini Starmap Footprint (P1)**
   - Draw rectangle footprint showing matched image region on the mini starmap for the current match.
   - Sync with match navigation; ensure projection alignment with existing alt-az view.
8. **Partial Results Streaming (P1)**
   - Render the viewer using the latest available stage output; swap backgrounds/overlays as flags flip to ready.
   - Avoid blocking downstream steps; continue processing in background while earlier layers are viewable.
9. **Resilience & Performance (P2)**
   - Add guards for missing/failed stages; keep viewer usable with fallback backgrounds.
   - Consider caching generated backgrounds/overlays in session to prevent rework when toggles change.
10. **Testing & UX Polish (P2)**
    - Smoke tests: upload sample, verify auto background generation, detection, matching, layer gating.
    - Visual QA: compact layout, minimal text, sidebar-only processing controls, main-area layer toggles, status visibility.

## Task Breakdown by File

- `streamlit_app.py`: Layout refactor, state flags, auto pipeline triggers, layer gating logic, viewer rendering updates, mini starmap footprint overlay, status display.
- `src/celebration_constellation/visualization.py`: Helpers for footprint rectangle and any new composed layers (if needed); ensure background builders accept staged inputs.
- `src/celebration_constellation/detection.py`: No major changes expected; ensure callable in auto pipeline; minor tweaks to expose progress hooks if needed.
- `src/celebration_constellation/matching.py` & `astronomy.py`: Ensure matching callable without manual CTA; optionally expose incremental status callbacks.

## Acceptance Checklist

- Upload immediately displays the image; pipeline starts without button presses.
- Background selector starts with black/synthwave baseline; dark greyscale and synthwave-from-image appear automatically when ready.
- Circle layer toggle becomes active only after detection completes; constellation overlays/star-color modes only after matches complete.
- Single top viewer contains all layer toggles; sidebar only holds processing parameters.
- Mini starmap shows matched footprint rectangle aligned to selected match.
- Interface remains compact with reduced explanatory text; progress/status is visible but minimal.
- Processing parameters are applied via a single Run button; no auto-rerun on parameter tweaks.
- Buttons that use icons rely on Font Awesome (no emoji) and do not mix icon + text on the same control.

"""Render a self-contained HTML labeling UI for one plan.

The output is a single .html file that:
  - Embeds the rendered PDF page as a base64 PNG (no external assets)
  - Overlays clickable SVG polygons for each detected room
  - Has a sidebar for setting correct_type per room + triaging cross-references
  - Produces a downloadable YAML compatible with
    scripts/eval_coordinator_tasks.py

Open the HTML in any browser — no server, no Python dependencies beyond this
generation step.

Usage:
    python scripts/render_labeling_ui.py <pipeline_output.json> <source_pdf>
                                         [--page N] [--out PATH] [--dpi N]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import fitz


# Must stay in sync with scripts/eval_coordinator_tasks.ALLOWED_ROOM_TYPES
ALLOWED_ROOM_TYPES = [
    "unit", "bathroom", "kitchen", "bedroom", "living_room", "dining_room",
    "closet", "entry", "garage",
    "stair", "hallway", "mechanical", "laundry", "storage", "office",
    "unknown",
]
WRONG_DETECTION = "wrong_detection"
TODO_SENTINEL = "TODO"

# Colors used for polygon fill per correct_type. Hex strings for direct CSS.
ROOM_TYPE_FILL = {
    "TODO":            "rgba(255, 193, 7, 0.30)",    # amber
    WRONG_DETECTION:   "rgba(244, 67, 54, 0.30)",    # red
    "unit":            "rgba(76, 175, 80, 0.28)",    # green
    "bathroom":        "rgba(33, 150, 243, 0.28)",   # blue
    "kitchen":         "rgba(255, 152, 0, 0.28)",    # orange
    "bedroom":         "rgba(103, 58, 183, 0.26)",   # deep purple
    "living_room":     "rgba(139, 195, 74, 0.28)",   # light green
    "dining_room":     "rgba(233, 30, 99, 0.24)",    # pink
    "closet":          "rgba(161, 136, 127, 0.28)",  # taupe
    "entry":           "rgba(255, 235, 59, 0.28)",   # yellow
    "garage":          "rgba(84, 110, 122, 0.30)",   # slate
    "stair":           "rgba(156, 39, 176, 0.28)",   # purple
    "hallway":         "rgba(205, 220, 57, 0.30)",   # lime
    "mechanical":      "rgba(96, 125, 139, 0.28)",   # blue-grey
    "laundry":         "rgba(0, 188, 212, 0.28)",    # cyan
    "storage":         "rgba(121, 85, 72, 0.28)",    # brown
    "office":          "rgba(0, 150, 136, 0.28)",    # teal
    "unknown":         "rgba(158, 158, 158, 0.22)",  # neutral grey
}


class _PageGeometry:
    """Everything needed to map pipeline schema coords to rendered-PNG pixels.

    The pipeline's extract_paths flips y using `page.rect.height` on points
    that PyMuPDF returned in MediaBox (pre-rotation) space. That mixes two
    coord systems whenever the PDF has a non-zero /Rotate flag. To paint an
    overlay that aligns with the rendered PNG we have to:

      1. Inverse the pipeline's flip (schema → mediabox coords)
      2. Apply the MediaBox→rect rotation transform
      3. Scale rect coords to image pixels

    This class centralises all four dimensions and the rotation so the
    conversion is a single clean function.
    """

    def __init__(
        self,
        mediabox_width: float,
        mediabox_height: float,
        rect_width: float,
        rect_height: float,
        rotation: int,
        image_width: float,
        image_height: float,
    ) -> None:
        self.mbw = mediabox_width
        self.mbh = mediabox_height
        self.rw = rect_width
        self.rh = rect_height
        self.rotation = rotation % 360
        self.iw = image_width
        self.ih = image_height

    def to_image_px(self, schema_x: float, schema_y: float) -> tuple[float, float]:
        # Step 1: recover native mediabox coords (inverse of pipeline flip,
        # which used page.rect.height — we follow the same convention).
        mbx = schema_x
        mby = self.rh - schema_y

        # Step 2: rotate mediabox point into rect (both top-left origin).
        if self.rotation == 0:
            rx, ry = mbx, mby
        elif self.rotation == 90:
            rx, ry = self.mbh - mby, mbx
        elif self.rotation == 180:
            rx, ry = self.mbw - mbx, self.mbh - mby
        elif self.rotation == 270:
            rx, ry = mby, self.mbw - mbx
        else:
            # Unusual rotation (e.g. 45°) isn't supported by PDF/A anyway.
            raise ValueError(f"Unsupported /Rotate={self.rotation}; expected 0/90/180/270.")

        # Step 3: scale to image pixels.
        return rx * (self.iw / self.rw), ry * (self.ih / self.rh)


def _render_page_as_png(source_pdf: Path, page_num: int, dpi: int) -> tuple[bytes, _PageGeometry]:
    """Render page to PNG and return the geometry needed for overlay mapping."""
    doc = fitz.open(source_pdf)
    try:
        if page_num < 0 or page_num >= doc.page_count:
            raise ValueError(f"--page {page_num} out of range [0, {doc.page_count - 1}]")
        page = doc[page_num]
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        geom = _PageGeometry(
            mediabox_width=float(page.mediabox.width),
            mediabox_height=float(page.mediabox.height),
            rect_width=float(page.rect.width),
            rect_height=float(page.rect.height),
            rotation=int(page.rotation),
            image_width=float(pix.width),
            image_height=float(pix.height),
        )
        return pix.tobytes("png"), geom
    finally:
        doc.close()


def _room_polygon_to_svg(polygon: list[list[float]], geom: _PageGeometry) -> str:
    """Convert a room polygon from schema coords to SVG viewBox coords
    matching the embedded PNG, accounting for PDF rotation."""
    if not polygon:
        return ""
    points = []
    for pt in polygon:
        px, py = geom.to_image_px(float(pt[0]), float(pt[1]))
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def build_html(doc: dict, png_bytes: bytes, geom: _PageGeometry) -> str:
    # Pre-compute SVG-ready polygon strings keyed by room_id.
    rooms_for_js = []
    for room in (doc.get("rooms") or []):
        rooms_for_js.append({
            "room_id": room.get("room_id"),
            "centroid": room.get("centroid") or [0, 0],
            "area": room.get("area") or 0.0,
            "detected_type": room.get("room_type") or "unknown",
            "room_name": room.get("room_name"),
            "room_number": room.get("room_number"),
            "svg_points": _room_polygon_to_svg(room.get("polygon") or [], geom),
        })

    xref_targets = sorted({
        x.get("target_sheet")
        for x in (doc.get("cross_references") or [])
        if x.get("target_sheet")
    })

    b64 = base64.b64encode(png_bytes).decode("ascii")

    context = {
        "plan_stem": doc.get("metadata", {}).get("source_pdf") or "",
        "source_pdf": doc.get("metadata", {}).get("source_pdf"),
        "pipeline_version": doc.get("metadata", {}).get("pipeline_version"),
        "image_width": geom.iw,
        "image_height": geom.ih,
        "pdf_rotation": geom.rotation,
        "image_b64": b64,
        "rooms": rooms_for_js,
        "xref_targets": list(xref_targets),
        "allowed_types": ALLOWED_ROOM_TYPES,
        "wrong_detection": WRONG_DETECTION,
        "todo": TODO_SENTINEL,
        "fills": ROOM_TYPE_FILL,
    }

    return _HTML_TEMPLATE.replace(
        "__CONTEXT_JSON__",
        json.dumps(context, separators=(",", ":")),
    )


# The HTML is one self-contained page. All runtime logic lives in the embedded
# <script>. CONTEXT_JSON is injected by build_html() above.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Labeling UI</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, system-ui, Segoe UI, sans-serif; }
  body { display: flex; }
  #canvas { flex: 1; overflow: auto; background: #2a2a2a; }
  #canvas svg { display: block; margin: 8px; box-shadow: 0 0 12px #000a; background: white; }
  #sidebar { width: 440px; border-left: 1px solid #ccc; background: #f7f7f7; display: flex; flex-direction: column; }
  #sidebar header { padding: 12px 14px; border-bottom: 1px solid #ddd; background: #fff; }
  #sidebar header h1 { font-size: 14px; margin: 0 0 2px 0; word-break: break-all; }
  #sidebar header .meta { font-size: 11px; color: #666; }
  #sidebar .scroll { flex: 1; overflow-y: auto; padding: 10px 14px; }
  #sidebar .section { margin-bottom: 18px; }
  #sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #555; margin: 0 0 8px 0; }
  .room-row { display: flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 4px; margin-bottom: 3px; font-size: 12px; cursor: pointer; }
  .room-row:hover { background: #e9e9e9; }
  .room-row.selected { background: #d8ecff; outline: 1px solid #2196f3; }
  .room-row .chip { display: inline-block; width: 10px; height: 10px; border-radius: 2px; border: 1px solid #aaa; flex: 0 0 auto; }
  .room-row .meta { color: #777; font-size: 10px; }
  .room-row select { font-size: 11px; padding: 2px 3px; }
  .room-row .name { flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .sheet-row { display: flex; align-items: center; gap: 6px; font-size: 12px; padding: 3px 6px; }
  .sheet-row code { background: #fff; padding: 1px 4px; border-radius: 3px; border: 1px solid #ddd; font-size: 11px; }
  #sidebar textarea { width: 100%; font-family: monospace; font-size: 11px; box-sizing: border-box; min-height: 60px; }
  #sidebar footer { padding: 10px 14px; border-top: 1px solid #ddd; background: #fff; display: flex; gap: 8px; }
  button { padding: 7px 12px; font-size: 12px; border: 1px solid #888; background: #fafafa; cursor: pointer; border-radius: 3px; }
  button.primary { background: #2196f3; color: white; border-color: #1976d2; }
  button:hover { filter: brightness(0.96); }
  .progress { font-size: 11px; color: #444; padding: 4px 0; }
  polygon.room { stroke: #333; stroke-width: 1.4; cursor: pointer; }
  polygon.room:hover { stroke: #000; stroke-width: 2.4; }
  polygon.room.selected { stroke: #2196f3; stroke-width: 3; }
  #missed-list .missed-row { display: flex; gap: 6px; align-items: center; font-size: 12px; padding: 3px 0; }
  #missed-list input { width: 60px; font-size: 11px; }
  .label-box { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; font-size: 11px; color: #555; }
  details summary { cursor: pointer; font-weight: bold; font-size: 12px; }
  details[open] summary { margin-bottom: 6px; }
  .help { font-size: 11px; color: #666; background: #fffbe6; border: 1px solid #f4d793; padding: 6px 8px; border-radius: 4px; margin-bottom: 10px; }
</style>
</head>
<body>
<div id="canvas">
  <svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
<div id="sidebar">
  <header>
    <h1 id="plan-title"></h1>
    <div class="meta" id="plan-meta"></div>
  </header>
  <div class="scroll">
    <div class="help">
      Click a room polygon to select it (scrolls the list to match). Set
      <code>correct_type</code> from the dropdown next to each row. Rooms left as
      <code>TODO</code> will block the evaluation from running. Use
      <b>Load&nbsp;YAML</b> to resume a previous session.
    </div>
    <div class="progress" id="progress"></div>
    <div class="section">
      <h2>Rooms</h2>
      <div id="room-list"></div>
    </div>
    <div class="section">
      <h2>Missed rooms</h2>
      <details>
        <summary>Add missed room</summary>
        <div id="missed-list"></div>
        <button id="add-missed">+ Add row</button>
      </details>
    </div>
    <div class="section">
      <h2>Referenced sheets</h2>
      <div class="label-box"><span>Check = valid  ·  Uncheck = noise</span><span id="xref-count"></span></div>
      <div id="sheet-list"></div>
      <h2 style="margin-top: 10px;">Missed targets</h2>
      <textarea id="missed-targets" placeholder="One per line (e.g. A402)"></textarea>
    </div>
  </div>
  <footer>
    <input type="file" id="load-yaml" accept=".yaml,.yml" style="display:none">
    <button id="load-yaml-btn">Load YAML</button>
    <button id="download" class="primary">Download YAML</button>
  </footer>
</div>

<script>
const CTX = __CONTEXT_JSON__;

// ------------------------------------------------------------- state
const state = {
  rooms: CTX.rooms.map(r => ({...r, correct_type: CTX.todo})),
  missed_rooms: [],
  xref_valid: new Set(CTX.xref_targets),  // all start valid
  missed_targets: "",
  selectedRoomId: null,
};

// ------------------------------------------------------------- rendering
function renderSvg() {
  const svg = document.getElementById('svg');
  svg.setAttribute('viewBox', `0 0 ${CTX.image_width} ${CTX.image_height}`);
  svg.setAttribute('width', CTX.image_width);
  svg.setAttribute('height', CTX.image_height);
  svg.innerHTML = '';
  const img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
  img.setAttribute('href', 'data:image/png;base64,' + CTX.image_b64);
  img.setAttribute('x', 0); img.setAttribute('y', 0);
  img.setAttribute('width', CTX.image_width);
  img.setAttribute('height', CTX.image_height);
  svg.appendChild(img);
  const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  for (const room of state.rooms) {
    if (!room.svg_points) continue;
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('class', 'room');
    poly.setAttribute('points', room.svg_points);
    poly.setAttribute('data-room-id', room.room_id);
    poly.style.fill = CTX.fills[room.correct_type] || CTX.fills[CTX.todo];
    if (state.selectedRoomId === room.room_id) poly.classList.add('selected');
    poly.addEventListener('click', () => selectRoom(room.room_id, true));
    group.appendChild(poly);
  }
  svg.appendChild(group);
}

function roomTypes() {
  return [CTX.todo, CTX.wrong_detection, ...CTX.allowed_types];
}

function renderSidebar() {
  document.getElementById('plan-title').textContent = CTX.plan_stem || '(no name)';
  document.getElementById('plan-meta').textContent =
    `v${CTX.pipeline_version} · rooms=${CTX.rooms.length} · xrefs=${CTX.xref_targets.length}`;

  // Progress line
  const todoCount = state.rooms.filter(r => r.correct_type === CTX.todo).length;
  const labeledCount = state.rooms.length - todoCount;
  document.getElementById('progress').innerHTML =
    `<b>${labeledCount}/${state.rooms.length}</b> rooms labeled · <b>${todoCount}</b> remaining`;

  // Rooms list
  const rl = document.getElementById('room-list');
  rl.innerHTML = '';
  for (const room of state.rooms) {
    const row = document.createElement('div');
    row.className = 'room-row';
    row.classList.add(room.correct_type === CTX.todo ? 'TODO' : '');
    if (state.selectedRoomId === room.room_id) row.classList.add('selected');
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.background = CTX.fills[room.correct_type] || CTX.fills[CTX.todo];
    const label = document.createElement('span');
    label.className = 'name';
    const id8 = (room.room_id || '').slice(0, 8);
    const name = room.room_name ? ` · ${room.room_name}` : '';
    const num = room.room_number ? ` · #${room.room_number}` : '';
    label.textContent = `${id8}${num}${name}  (detected: ${room.detected_type})`;
    const select = document.createElement('select');
    for (const t of roomTypes()) {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      if (t === room.correct_type) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener('change', (e) => {
      room.correct_type = e.target.value;
      renderSvg();
      renderSidebar();
    });
    row.addEventListener('click', (e) => {
      if (e.target === select || e.target.tagName === 'OPTION') return;
      selectRoom(room.room_id, false);
    });
    row.appendChild(chip);
    row.appendChild(label);
    row.appendChild(select);
    rl.appendChild(row);
  }

  // Missed rooms
  const ml = document.getElementById('missed-list');
  ml.innerHTML = '';
  state.missed_rooms.forEach((m, idx) => {
    const row = document.createElement('div');
    row.className = 'missed-row';
    const x = document.createElement('input');
    x.placeholder = 'x'; x.value = m.centroid[0]; x.addEventListener('input', (e) => {
      m.centroid[0] = parseFloat(e.target.value) || 0;
    });
    const y = document.createElement('input');
    y.placeholder = 'y'; y.value = m.centroid[1]; y.addEventListener('input', (e) => {
      m.centroid[1] = parseFloat(e.target.value) || 0;
    });
    const s = document.createElement('select');
    for (const t of CTX.allowed_types) {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      if (t === m.correct_type) opt.selected = true;
      s.appendChild(opt);
    }
    s.addEventListener('change', (e) => { m.correct_type = e.target.value; });
    const del = document.createElement('button');
    del.textContent = '×';
    del.style.padding = '0 8px';
    del.addEventListener('click', () => {
      state.missed_rooms.splice(idx, 1);
      renderSidebar();
    });
    row.appendChild(x); row.appendChild(y); row.appendChild(s); row.appendChild(del);
    ml.appendChild(row);
  });

  // Cross-references
  const sl = document.getElementById('sheet-list');
  sl.innerHTML = '';
  document.getElementById('xref-count').textContent =
    `${state.xref_valid.size} / ${CTX.xref_targets.length} kept`;
  for (const tgt of CTX.xref_targets) {
    const row = document.createElement('div');
    row.className = 'sheet-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.xref_valid.has(tgt);
    cb.addEventListener('change', () => {
      if (cb.checked) state.xref_valid.add(tgt);
      else state.xref_valid.delete(tgt);
      renderSidebar();
    });
    const code = document.createElement('code');
    code.textContent = tgt;
    row.appendChild(cb); row.appendChild(code);
    sl.appendChild(row);
  }

  document.getElementById('missed-targets').value = state.missed_targets;
}

// ------------------------------------------------------------- interactions
function selectRoom(room_id, scrollList) {
  state.selectedRoomId = room_id;
  renderSvg();
  renderSidebar();
  if (scrollList) {
    const rows = document.querySelectorAll('.room-row.selected');
    if (rows[0]) rows[0].scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

// ------------------------------------------------------------- YAML serialization
function yamlEscape(v) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  const s = String(v);
  if (/^[A-Za-z0-9_\-.]+$/.test(s)) return s;
  return JSON.stringify(s);  // JSON strings are valid YAML strings
}

function toYaml() {
  const lines = [];
  lines.push(`plan_stem: ${yamlEscape(CTX.plan_stem)}`);
  lines.push(`source_pdf: ${yamlEscape(CTX.source_pdf)}`);
  lines.push(`pipeline_version: ${yamlEscape(CTX.pipeline_version)}`);
  lines.push('rooms:');
  for (const r of state.rooms) {
    lines.push(`- room_id: ${yamlEscape(r.room_id)}`);
    lines.push(`  centroid:`);
    lines.push(`  - ${r.centroid[0]}`);
    lines.push(`  - ${r.centroid[1]}`);
    lines.push(`  area: ${r.area}`);
    lines.push(`  detected_type: ${yamlEscape(r.detected_type)}`);
    lines.push(`  room_name: ${yamlEscape(r.room_name)}`);
    lines.push(`  room_number: ${yamlEscape(r.room_number)}`);
    lines.push(`  correct_type: ${yamlEscape(r.correct_type)}`);
  }
  lines.push('missed_rooms:');
  for (const m of state.missed_rooms) {
    lines.push(`- centroid:`);
    lines.push(`  - ${m.centroid[0]}`);
    lines.push(`  - ${m.centroid[1]}`);
    lines.push(`  correct_type: ${yamlEscape(m.correct_type)}`);
  }
  lines.push('cross_references:');
  lines.push('  detected_targets:');
  for (const t of CTX.xref_targets) lines.push(`  - ${yamlEscape(t)}`);
  lines.push('  valid_targets:');
  for (const t of CTX.xref_targets) if (state.xref_valid.has(t)) lines.push(`  - ${yamlEscape(t)}`);
  const missed = (state.missed_targets || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  lines.push('  missed_targets:');
  for (const t of missed) lines.push(`  - ${yamlEscape(t)}`);
  return lines.join('\n') + '\n';
}

function downloadYaml() {
  const yaml = toYaml();
  const blob = new Blob([yaml], {type: 'text/yaml'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const safeStem = (CTX.plan_stem || 'labels').replace(/[^A-Za-z0-9_\-]+/g, '_');
  a.download = `${safeStem}.labels.yaml`;
  a.click();
  URL.revokeObjectURL(url);
}

// Minimal YAML loader — just enough to round-trip the template format
function parseYaml(text) {
  const lines = text.split(/\r?\n/);
  const result = {rooms: [], missed_rooms: [], cross_references: {detected_targets: [], valid_targets: [], missed_targets: []}};
  let section = null;
  let subsection = null;
  let current = null;
  for (const raw of lines) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    const m = raw.match(/^(\s*)(- )?(\w+)?:?\s*(.*)$/);
    if (!m) continue;
    const indent = m[1].length;
    if (indent === 0) {
      if (raw.startsWith('rooms:')) { section = 'rooms'; current = null; }
      else if (raw.startsWith('missed_rooms:')) { section = 'missed_rooms'; current = null; }
      else if (raw.startsWith('cross_references:')) { section = 'cross_references'; subsection = null; }
      continue;
    }
    if (section === 'cross_references' && indent >= 2) {
      if (raw.match(/^\s*detected_targets:/)) subsection = 'detected';
      else if (raw.match(/^\s*valid_targets:/)) subsection = 'valid';
      else if (raw.match(/^\s*missed_targets:/)) subsection = 'missed';
      else if (raw.match(/^\s*-\s+/)) {
        const val = raw.trim().replace(/^-\s+/, '').replace(/^"|"$/g, '');
        if (subsection === 'valid') result.cross_references.valid_targets.push(val);
        else if (subsection === 'missed') result.cross_references.missed_targets.push(val);
      }
      continue;
    }
    if (section === 'rooms' || section === 'missed_rooms') {
      if (raw.match(/^-\s+/)) {
        current = {};
        (section === 'rooms' ? result.rooms : result.missed_rooms).push(current);
      }
      const kv = raw.match(/^\s*-?\s*(\w+):\s*(.*)$/);
      if (kv && current) {
        const k = kv[1], v = kv[2].trim();
        if (k === 'centroid') current.centroid = [];
        else if (v.match(/^-?\d+(\.\d+)?$/)) current[k] = parseFloat(v);
        else if (v === 'null' || v === '') current[k] = null;
        else current[k] = v.replace(/^"|"$/g, '');
      }
      // centroid list items
      if (current && current.centroid !== undefined && raw.match(/^\s*-\s+-?\d/)) {
        current.centroid.push(parseFloat(raw.trim().slice(2)));
      }
    }
  }
  return result;
}

function loadYamlFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = parseYaml(e.target.result);
      // apply to state
      const byId = new Map(state.rooms.map(r => [r.room_id, r]));
      for (const r of parsed.rooms || []) {
        if (byId.has(r.room_id) && r.correct_type) {
          byId.get(r.room_id).correct_type = r.correct_type;
        }
      }
      state.missed_rooms = (parsed.missed_rooms || []).map(r => ({
        centroid: r.centroid || [0, 0],
        correct_type: r.correct_type || 'unknown',
      }));
      state.xref_valid = new Set(parsed.cross_references?.valid_targets || []);
      state.missed_targets = (parsed.cross_references?.missed_targets || []).join('\n');
      renderSvg();
      renderSidebar();
      alert('Loaded labels from YAML.');
    } catch (err) {
      alert('Failed to parse YAML: ' + err.message);
    }
  };
  reader.readAsText(file);
}

// ------------------------------------------------------------- init
document.addEventListener('DOMContentLoaded', () => {
  renderSvg();
  renderSidebar();
  document.getElementById('download').addEventListener('click', downloadYaml);
  document.getElementById('add-missed').addEventListener('click', () => {
    state.missed_rooms.push({centroid: [0, 0], correct_type: 'unknown'});
    renderSidebar();
  });
  document.getElementById('missed-targets').addEventListener('input', (e) => {
    state.missed_targets = e.target.value;
  });
  document.getElementById('load-yaml-btn').addEventListener('click', () => {
    document.getElementById('load-yaml').click();
  });
  document.getElementById('load-yaml').addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) loadYamlFile(e.target.files[0]);
  });
});
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a self-contained HTML labeling UI for one plan.")
    parser.add_argument("pipeline_output", type=Path)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("--page", type=int, default=0, help="0-indexed page (default: 0).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=120, help="Render DPI for the embedded PNG (default: 120).")
    args = parser.parse_args(argv)

    with args.pipeline_output.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    png_bytes, geom = _render_page_as_png(args.source_pdf, args.page, args.dpi)
    html = build_html(doc, png_bytes, geom)

    out_path = args.out or Path("output") / "labeling_ui" / f"{args.pipeline_output.stem}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote labeling UI: {out_path}")
    print(f"  size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"  rooms: {len(doc.get('rooms') or [])}")
    print(f"  xref targets: {len({x.get('target_sheet') for x in (doc.get('cross_references') or []) if x.get('target_sheet')})}")
    print(f"  Open it in any browser to start labeling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

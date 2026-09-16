from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import anthropic
import fitz
import os
import json
import base64
import asyncio
import concurrent.futures
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PAGE SCORING — TIERED ──
TIER_1_KEYWORDS = [
    'division 03', 'div 03', 'div. 03',
    '03 30 00', '03 35 00', '03 05 00', '03300', '03350', '03050',
    'cast-in-place concrete', 'cast in place concrete',
    'section 03', 'spec section 03',
    'w/c ratio', 'water-cement ratio', 'water cement ratio',
    'fly ash percentage', 'fly ash replacement', 'flyash replacement',
    'max w/c', 'maximum w/c', 'cement type',
    'air content', 'slump requirement',
    'admixture brand', 'admixture spec',
    'fiber reinforcement', 'fibermesh', 'synthetic fiber',
    'chloride restriction', 'sulfate exposure', 'sulfate class',
    'cure spec', 'curing requirement', 'testing frequency',
]

TIER_2_KEYWORDS = [
    's0.0', 's0.1', 's-0.0', 's-0.1', 's1.0', 's1.1', 's-1.0',
    'sheet s0', 'sheet s-0', 'sheet s1',
    'structural general note', 'general structural note',
    'general notes', 'structural notes',
    'concrete schedule', 'concrete mix schedule', 'mix schedule',
    'concrete strength', 'minimum concrete strength',
    'building component', 'compressive strength',
    'normal weight', 'lightweight concrete',
    'footing', 'slab on grade', 'slab-on-grade',
    'grade beam', 'drilled pier', 'drilled shaft',
    'exposure class', 'exposure category',
    'no fly ash', 'fly ash not permitted', 'fly ash prohibited',
    'non-chloride', 'aci 318', 'aci 301', 'aci 211',
    "f'c =", "f'c=", 'fc =', 'fc=',
    '3000 psi', '3500 psi', '4000 psi', '4500 psi', '5000 psi',
    '3,000 psi', '3,500 psi', '4,000 psi', '4,500 psi', '5,000 psi',
    'pounds per square inch', 'psi)', 'w/c ratio', 'slump',
    'aggregate size', 'max aggregate',
]

TIER_3_KEYWORDS = [
    'c0.', 'c-0.', 'c5.', 'c-5.', 'sheet c0', 'sheet c5',
    'pavement detail', 'paving detail', 'pavement section',
    'item 421', 'item421', 'txdot item',
    'class p concrete', 'class p pavement',
    'fiber reinforced pavement',
    '6 sack', '6-sack', 'six sack',
    'pavement thickness', 'subgrade preparation',
]

CONCRETE_KEYWORDS = [
    'concrete', 'psi', "f'c", 'fc=', 'mix design', 'ready mix', 'ready-mix',
    'cubic yard', 'cubic yds', 'c.y.', ' cy ', 'sack', 'cement',
    'reinforc', 'footing', 'foundation', 'slab', 'grade beam', 'pile cap',
    'drilled shaft', 'pier', 'retaining wall', 'curb', 'gutter',
    'sidewalk', 'pavement', 'flatwork', 'pour', 'placement', 'admixture',
    'fly ash', 'flyash', 'slag', 'silica fume', 'air entrain', 'water cement',
    'slump', 'fibermesh', 'fiber mesh', 'superplastic', 'accelerat', 'retard',
    'compressive strength', '3000', '3500', '4000', '4500', '5000', '6000',
    'w/c', 'portland', 'structural concrete', 'cast-in-place', 'reinforced',
    'anchor bolt', 'pedestal', 'column', 'spread footing', 'mat slab',
    'rebar', 'dowel', 'embed',
]

SKIP_KEYWORDS = [
    'plumbing fixture', 'hvac unit', 'ductwork schedule', 'electrical panel schedule',
    'lighting fixture schedule', 'sprinkler head schedule',
    'landscape planting', 'irrigation head',
    'paint color schedule', 'carpet schedule',
    'acoustical ceiling', 'door hardware schedule',
    'curtain wall detail', 'drywall partition',
    'roofing membrane detail',
    'boring log', 'soil boring', 'geotechnical report',
    'atterberg limit', 'plasticity index',
    'toilet partition', 'fire alarm device',
    'panel schedule', 'load calculation', 'photometric',
]

# ── VALID CONCRETE ELEMENT ANCHORS FOR VALIDATOR ──
VALID_CONCRETE_ANCHORS = [
    'slab', 'footing', 'foundation', 'grade beam', 'column', 'wall',
    'pavement', 'curb', 'gutter', 'sidewalk', 'flatwork', 'pier',
    'drilled shaft', 'pile cap', 'retaining', 'mat slab', 'topping',
    'elevated slab', 'structural slab', 'podium', 'tilt', 'panel',
    'cast-in-place', 'cast in place', 'concrete', 'placement',
    'division 03', '03 30 00', 'spec', 'schedule', 'all other',
    'unassigned',
]

# ── INVALID SOURCE INDICATORS — TRADE ISOLATION ──
INVALID_SOURCES = [
    'rebar', 'bar size', '#3', '#4', '#5', '#6', '#7', '#8',
    'development length', 'ld for', 'lap splice', 'splice length',
    'structural steel', 'steel beam', 'wide flange', 'hss',
    'masonry', 'cmu', 'mortar', 'grout unit',
    'mechanical', 'plumbing', 'electrical',
]


def clean_extracted_text(raw_text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', raw_text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_table_heavy_page(text: str, page_width: float, page_height: float) -> bool:
    text_lower = text.lower()
    text_len = len(text.strip())
    page_area = page_width * page_height
    text_density = text_len / max(page_area, 1)
    schedule_signals = [
        'building component', 'compressive strength', 'normal weight',
        'pounds per square inch', 'psi)', 'max aggregate',
        'w/c ratio', '28 day', '28-day', 'cylinder',
        'footing', 'slab', 'grade beam',
    ]
    schedule_hits = sum(1 for s in schedule_signals if s in text_lower)
    if schedule_hits >= 2 and text_density < 0.01:
        return True
    psi_values = len(re.findall(r'\b(3000|3500|4000|4500|5000|6000)\b', text))
    if psi_values >= 2 and text_len < 500:
        return True
    return False


def score_page(text: str) -> tuple[int, str]:
    text_lower = text.lower()
    if len(text_lower.strip()) < 30:
        return 0, "blank/image-only page"
    skip_hits = [kw for kw in SKIP_KEYWORDS if kw in text_lower]
    if skip_hits:
        concrete_hits = sum(1 for k in CONCRETE_KEYWORDS[:20] if k in text_lower)
        if concrete_hits < 2:
            return 0, f"non-concrete ({skip_hits[0]})"
    tier1_hits = [kw for kw in TIER_1_KEYWORDS if kw in text_lower]
    if tier1_hits:
        return 1, f"Div03/specs ({tier1_hits[0]})"
    tier2_hits = [kw for kw in TIER_2_KEYWORDS if kw in text_lower]
    if tier2_hits:
        return 2, f"structural notes ({tier2_hits[0]})"
    tier3_hits = [kw for kw in TIER_3_KEYWORDS if kw in text_lower]
    if tier3_hits:
        return 3, f"civil/paving ({tier3_hits[0]})"
    concrete_hits = [kw for kw in CONCRETE_KEYWORDS if kw in text_lower]
    if len(concrete_hits) >= 2:
        return 4, f"concrete content ({', '.join(concrete_hits[:3])})"
    strong_signals = ["f'c", 'fc=', 'psi', 'concrete', 'mix design', 'admixture', 'fly ash', 'fibermesh']
    strong_hits = [s for s in strong_signals if s in text_lower]
    if len(strong_hits) >= 1 and len(text_lower) < 2000:
        return 4, f"short page with concrete signal ({strong_hits[0]})"
    return 0, "no concrete keywords found"


def page_to_image_b64(page, scale: float = 2.0) -> str:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode()


def filter_and_chunk_pdf(pdf_bytes: bytes, filename: str) -> tuple[list, dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    scored_pages = []
    skipped_pages = []
    image_render_pages = {}

    for i in range(total_pages):
        page = doc[i]
        raw_text = page.get_text()
        text = clean_extracted_text(raw_text)
        tier, reason = score_page(text)
        if tier > 0:
            scored_pages.append((tier, i, reason))
            rect = page.rect
            if is_table_heavy_page(text, rect.width, rect.height):
                print(f"  Page {i+1}: TABLE-HEAVY — rendering as image")
                image_render_pages[i] = page_to_image_b64(page, scale=2.0)
            else:
                print(f"  Page {i+1}: Tier {tier} — {reason}")
        else:
            skipped_pages.append((i + 1, reason))
            print(f"  Page {i+1}: SKIP — {reason}")

    scored_pages.sort(key=lambda x: x[0])
    concrete_page_indices = [p[1] for p in scored_pages]
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for tier, _, _ in scored_pages:
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"{filename}: {total_pages} pages, {len(concrete_page_indices)} concrete, {len(image_render_pages)} table-heavy")

    if not concrete_page_indices:
        doc.close()
        return [], {
            "total_pages": total_pages, "concrete_pages": 0,
            "skipped_pages": total_pages,
            "message": "No concrete pages found — try uploading structural or civil drawings"
        }

    MAX_CHUNK_BYTES = 5 * 1024 * 1024
    chunks = []
    chunk_size = 8

    for i in range(0, len(concrete_page_indices), chunk_size):
        page_indices = concrete_page_indices[i:i + chunk_size]
        new_doc = fitz.open()
        for pi in page_indices:
            new_doc.insert_pdf(doc, from_page=pi, to_page=pi)
        chunk_bytes = new_doc.tobytes(deflate=True, garbage=4, clean=True)
        new_doc.close()

        if len(chunk_bytes) > MAX_CHUNK_BYTES:
            new_doc2 = fitz.open()
            for pi in page_indices:
                page = doc[pi]
                for scale, quality in [(0.6, 55), (0.5, 45), (0.4, 35)]:
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("jpeg", jpg_quality=quality)
                    if len(img_bytes) < 400 * 1024:
                        break
                img_pdf = fitz.open("pdf", fitz.open("jpeg", img_bytes).convert_to_pdf())
                new_doc2.insert_pdf(img_pdf)
                img_pdf.close()
            chunk_bytes = new_doc2.tobytes(deflate=True, garbage=4)
            new_doc2.close()

        page_nums = [p + 1 for p in page_indices]
        chunk_images = {p + 1: image_render_pages[p] for p in page_indices if p in image_render_pages}
        chunks.append((chunk_bytes, page_nums, chunk_images))

    doc.close()
    return chunks, {
        "total_pages": total_pages,
        "concrete_pages": len(concrete_page_indices),
        "skipped_pages": len(skipped_pages),
        "chunks": len(chunks),
        "tier_breakdown": tier_counts,
        "table_pages": len(image_render_pages),
    }


# ── EXTRACTION PROMPT ──
EXTRACTION_PROMPT = """You are a construction specification extraction agent for a ready-mix concrete sales representative at JCK Batch Plant LLC in North Texas.

YOUR ONLY JOB: Extract concrete mix requirements needed for ready-mix ordering and pricing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH PRIORITY — in this order:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Division 03 specs (03 30 00, 03 35 00, 03 05 00)
2. Structural general notes (S0.0, S0.1, S1.0) — concrete schedules, PSI by element
3. Civil/pavement notes (C0.x, C5.x) — paving, TxDOT references
4. Foundation plans / footing schedules

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
READING CONCRETE SCHEDULE TABLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plans often show PSI requirements as a dot/bullet matrix:
  BUILDING COMPONENT | 3000 | 3500 | 4000 | MAX AGG | SLUMP | W/C
  Footings           |      |  ●   |      |  1"     | 5-7"  | 0.55
  Slab-on-Grade      |      |      |  ●   |  1"     | 4-6"  | 0.50

The FILLED DOT (●) position determines the PSI. Read it carefully.
Footings = 3500 PSI. Slab = 4000 PSI. Not 3000.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT EXCLUSIONS — NEVER EXTRACT FROM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Rebar development length tables (Ld charts) — COMPLETELY IGNORE
  These show bar sizes (#3,#4,#5) vs PSI columns with inch values
  The PSI column headers in Ld tables are NOT concrete specs — they are rebar calc inputs
  Even if 3000/4000/5000/6000 appear — if it is an Ld table, extract NOTHING from it
✗ Rebar lap splice tables
✗ Structural steel notes
✗ Masonry/CMU specs
✗ MEP/plumbing/electrical
✗ Geotechnical data
✗ Any PSI value not directly tied to a concrete placement element

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PSI VALIDITY RULE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A PSI value is ONLY valid if it is explicitly tied to a concrete element:
✓ "4000 PSI slab-on-grade"
✓ "f'c = 3500 psi for footings"
✓ Dot in the 3500 column next to "Footings" in a concrete schedule table
✗ "3000" appearing in an Ld table column header
✗ PSI with no element context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACT per concrete application:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- application: element name (Footings, Slab on Grade, Grade Beams, Columns, Pavement, etc.)
- psi: 28-day compressive strength — FROM CONCRETE SCHEDULE OR SPECS ONLY
- air_entrained: true/false
- cementitious: "20%FA" or "100%Cem"
- aggregate: "LS" limestone, "BL" blended
- wc_ratio: w/c ratio if stated
- fly_ash_limit: max fly ash % if stated
- no_fly_ash: true if explicitly prohibited
- fiber_type: "shrinkage" or "structural" or null
- fiber_lbs_per_cy: dosage if stated
- admixtures: list any special admixtures
- pump_required: true if pump mix specified
- early_strength_required: true if high early required
- early_strength_psi: e.g. "3000 @ 3 days"
- txdot_class: ONLY if "TxDOT Class X" or "Item 421" explicitly stated
- exposure_class: sulfate/freeze-thaw/chloride if stated
- aggregate_size: "1in", "3/4in", "3/8in" if specified
- slump: max slump if stated
- volume_cy: stated CY or null
- volume_estimated: true if calculated
- spec_from_plans: EXACT text from the plans that defines this requirement
- source_type: "concrete_schedule" | "division_03" | "structural_notes" | "civil_notes" | "inferred"
- notes: anything needing rep attention

TXDOT: Flag ONLY on explicit "TxDOT Item 421" or "TxDOT Class A/B/C/P/S/H"
NOT on FM road numbers, highway names, or project location

If pages contain no valid concrete specs, return empty concrete_requirements array.
Do NOT invent sack content.

Return ONLY valid JSON, no markdown:
{
  "project_name": "string or null",
  "client_name": "string or null",
  "client_contact": "string or null",
  "client_phone": "string or null",
  "client_email": "string or null",
  "project_address": "string or null",
  "bid_date": "string or null",
  "projected_start": "string or null",
  "project_type": "Commercial|Residential|TXDOT|Mixed",
  "bond_surety": "string or null",
  "payment_terms": "string or null",
  "total_yards_stated": null,
  "concrete_requirements": [
    {
      "application": "string",
      "spec_from_plans": "string",
      "source_type": "concrete_schedule|division_03|structural_notes|civil_notes|inferred",
      "psi": number,
      "air_entrained": boolean,
      "cementitious": "20%FA|100%Cem|other",
      "aggregate": "LS|BL|LP|PP|PGEXP|other",
      "wc_ratio": number or null,
      "fly_ash_limit": "string or null",
      "no_fly_ash": boolean,
      "fiber_type": "shrinkage|structural|null",
      "fiber_lbs_per_cy": number or null,
      "admixtures": ["string"],
      "pump_required": boolean,
      "early_strength_required": boolean,
      "early_strength_psi": "string or null",
      "txdot_class": "string or null",
      "exposure_class": "string or null",
      "aggregate_size": "string or null",
      "slump": "string or null",
      "volume_cy": number or null,
      "volume_estimated": boolean,
      "suggested_jck_code": "string",
      "notes": "string"
    }
  ],
  "addons": [
    {
      "name": "string",
      "unit": "CY|Load|Order|LF|SF",
      "rate": number or null,
      "qty": number or null,
      "notes": "string"
    }
  ],
  "flags": ["string"],
  "summary": "string"
}"""


# ── VALIDATOR PROMPT ──
VALIDATOR_PROMPT = """You are a bid safety validator for a ready-mix concrete sales operation.
Your job is to check extracted concrete specifications and remove any that are NOT safely attributable to concrete placements.
A wrong mix code costs thousands of dollars. When in doubt, exclude and flag.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION CHECKS — apply in order:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTEXT LOCK — Is the PSI tied to a concrete element?
   Valid: slab, footing, foundation, grade beam, column, wall, pavement, curb, pier, tilt panel
   Invalid: rebar table, steel schedule, load table, MEP note
   If source_type is "inferred" and application contains "Unassigned" → KEEP but flag for rep review
   If no valid concrete anchor → EXCLUDE

2. TRADE ISOLATION — Did this come from a non-concrete source?
   If spec_from_plans mentions: rebar sizes, bar marks, Ld values, development length,
   structural steel, masonry, plumbing → EXCLUDE

3. PSI RANGE CHECK — Is this a realistic concrete strength?
   Valid range: 2500–8000 PSI
   Out of range → FLAG but keep

4. DUPLICATE RESOLUTION — Same element, multiple PSI values?
   Prioritize: division_03 > structural_notes > concrete_schedule > civil_notes > inferred
   Keep highest-priority source, flag the conflict

5. CONSISTENCY CHECK — Does PSI match the element type?
   Residential slab < 4000 PSI is normal
   Commercial foundation > 5500 PSI without special note is suspicious → FLAG
   Never auto-reject on this — just flag

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return the same JSON structure as input
- Add "validation_status": "confirmed" | "flagged" | "excluded" to each requirement
- Add "validation_note": brief reason if flagged or excluded
- EXCLUDED items must still appear in the array with psi set to null so the rep sees them
- Add any new flags to the flags array
- Do NOT change psi, application, or spec_from_plans values — only annotate
- Return ONLY valid JSON, no markdown"""


def validate_requirements_sync(api_key: str, extracted: dict) -> dict:
    """
    Second Claude pass — validates extracted requirements using the bid safety rules.
    Removes Ld table contamination and flags anything suspicious.
    """
    if not extracted.get("concrete_requirements"):
        return extracted

    cl = anthropic.Anthropic(api_key=api_key)

    # Build a compact version of just the requirements for validation
    validation_input = {
        "concrete_requirements": extracted.get("concrete_requirements", []),
        "flags": extracted.get("flags", []),
    }

    print(f"  Validator: checking {len(validation_input['concrete_requirements'])} requirements...")

    try:
        resp = cl.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VALIDATOR_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"Validate these extracted concrete requirements:\n\n{json.dumps(validation_input, indent=2)}\n\nReturn the validated JSON."
                    }
                ]
            }]
        )

        raw = resp.content[0].text
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        validated = json.loads(cleaned)

        # Merge validated requirements back into the full result
        if "concrete_requirements" in validated:
            # Filter out excluded items (keep them for display but mark clearly)
            kept = []
            for req in validated["concrete_requirements"]:
                status = req.get("validation_status", "confirmed")
                if status == "excluded":
                    # Keep it visible but nulled out so rep knows it was found and rejected
                    req["psi"] = None
                    req["notes"] = f"EXCLUDED BY VALIDATOR: {req.get('validation_note', 'failed safety check')}"
                    kept.append(req)
                else:
                    kept.append(req)

            # Only return non-excluded for actual quoting
            extracted["concrete_requirements"] = [r for r in kept if r.get("validation_status") != "excluded"]
            excluded_count = len([r for r in kept if r.get("validation_status") == "excluded"])

            if excluded_count > 0:
                extracted["flags"].append(
                    f"Validator removed {excluded_count} requirement(s) that could not be safely tied to concrete elements — check Raw Extraction for details"
                )

        if "flags" in validated:
            for f in validated["flags"]:
                if f not in extracted["flags"]:
                    extracted["flags"].append(f)

        print(f"  Validator: {len(extracted['concrete_requirements'])} requirements passed, {excluded_count if 'excluded_count' in dir() else 0} excluded")
        return extracted

    except Exception as e:
        print(f"  Validator error (non-fatal): {str(e)[:100]}")
        extracted["flags"].append(f"Validator skipped due to error: {str(e)[:80]}")
        return extracted


def analyze_chunk_sync(api_key: str, chunk_bytes: bytes, page_nums: list,
                       filename: str, chunk_images: dict) -> dict:
    cl = anthropic.Anthropic(api_key=api_key)
    b64 = base64.b64encode(chunk_bytes).decode()
    chunk_mb = len(chunk_bytes) / (1024 * 1024)
    print(f"  Extracting pages {page_nums} ({chunk_mb:.2f}MB), {len(chunk_images)} image-rendered")

    content_blocks = [
        {
            "type": "text",
            "text": EXTRACTION_PROMPT,
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64
            },
            "title": f"{filename} p{page_nums[0]}-{page_nums[-1]}"
        }
    ]

    for page_num, img_b64 in chunk_images.items():
        content_blocks.append({
            "type": "text",
            "text": f"Page {page_num} as image — read dot/bullet positions in concrete schedule tables carefully:"
        })
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64
            }
        })

    content_blocks.append({
        "type": "text",
        "text": "Extract concrete mix requirements. For dot/bullet matrix tables, read bullet positions carefully — the dot column IS the PSI. Return only valid JSON."
    })

    try:
        resp = cl.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": content_blocks}]
        )
        raw = resp.content[0].text
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as je:
            print(f"  JSON truncated on pages {page_nums}, attempting repair...")
            last_good = cleaned.rfind('},', 0, je.pos)
            if last_good > 0:
                salvaged = cleaned[:last_good+1] + '],"addons":[],"flags":["WARNING: Response truncated — re-run this file"],"summary":""}'
                try:
                    return json.loads(salvaged)
                except:
                    pass
            return {
                "concrete_requirements": [], "addons": [],
                "flags": [f"Pages {page_nums}: response truncated — try re-running"]
            }

    except anthropic.BadRequestError as e:
        if "request_too_large" in str(e):
            return {"concrete_requirements": [], "addons": [], "flags": [f"Pages {page_nums} too large — skipped"]}
        return {"concrete_requirements": [], "addons": [], "flags": [f"Error pages {page_nums}: {str(e)[:100]}"]}
    except Exception as e:
        return {"concrete_requirements": [], "addons": [], "flags": [f"Error pages {page_nums}: {str(e)[:100]}"]}


async def analyze_chunks_parallel(chunks: list, filename: str, api_key: str) -> dict:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            loop.run_in_executor(
                executor, analyze_chunk_sync,
                api_key, chunk_bytes, page_nums, filename, chunk_images
            )
            for chunk_bytes, page_nums, chunk_images in chunks
        ]
        results = await asyncio.gather(*futures)
    merged = merge_results(list(results))

    # ── VALIDATOR PASS ──
    # Run synchronously after parallel extraction — checks all requirements
    # before returning to frontend
    if merged.get("concrete_requirements"):
        loop2 = asyncio.get_event_loop()
        merged = await loop2.run_in_executor(
            None, validate_requirements_sync, api_key, merged
        )

    return merged


def merge_results(results: list) -> dict:
    merged = {
        "project_name": None, "client_name": None, "client_contact": None,
        "client_phone": None, "client_email": None, "project_address": None,
        "bid_date": None, "projected_start": None, "project_type": None,
        "bond_surety": None, "payment_terms": None, "total_yards_stated": None,
        "concrete_requirements": [], "addons": [], "flags": [], "summary": ""
    }
    seen_apps = set()
    seen_addons = set()

    for r in results:
        for key in ["project_name", "client_name", "client_contact", "client_phone",
                    "client_email", "project_address", "bid_date", "projected_start",
                    "project_type", "bond_surety", "payment_terms", "total_yards_stated", "summary"]:
            if not merged[key] and r.get(key):
                merged[key] = r[key]

        for req in r.get("concrete_requirements", []):
            key = (req.get("application", "").lower().strip(), req.get("psi", 0))
            if key not in seen_apps:
                seen_apps.add(key)
                merged["concrete_requirements"].append(req)

        for addon in r.get("addons", []):
            key = addon.get("name", "").lower().strip()
            if key and key not in seen_addons:
                seen_addons.add(key)
                merged["addons"].append(addon)

        for flag in r.get("flags", []):
            if flag not in merged["flags"]:
                merged["flags"].append(flag)

    return merged


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="API key not configured")

    all_results = []

    for file in files:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        print(f"Processing {file.filename} ({size_mb:.1f}MB)")

        chunks, stats = filter_and_chunk_pdf(content, file.filename)

        if not chunks:
            all_results.append({
                "concrete_requirements": [], "addons": [],
                "flags": [f"{file.filename}: {stats.get('message', 'No concrete pages found')} ({stats.get('total_pages', 0)} pages scanned)"],
                "summary": ""
            })
            continue

        result = await analyze_chunks_parallel(chunks, file.filename, api_key)

        tier = stats.get("tier_breakdown", {})
        result["flags"].insert(0,
            f"{file.filename}: {stats['total_pages']} pages scanned, "
            f"{stats['concrete_pages']} concrete pages "
            f"(Div03={tier.get(1,0)}, StructNotes={tier.get(2,0)}, "
            f"Civil={tier.get(3,0)}, General={tier.get(4,0)}), "
            f"{stats.get('table_pages',0)} table-rendered, "
            f"{stats['chunks']} chunks extracted + validated"
        )
        all_results.append(result)

    return merge_results(all_results) if len(all_results) > 1 else all_results[0]


@app.get("/health")
def health():
    return {"status": "ok", "service": "Zamara Solutions Bid Analyzer v0.9"}

# ai_assist_risk.py
from __future__ import annotations
import json, re, time, logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


from typing import List
from pydantic import BaseModel, Field, ValidationError, confloat

from PIL import Image

# ---- HF Transformers (text LLM) ----
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# ---- Qwen VLM (document vision-language) ----
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except Exception:
    HAS_QWEN = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =============================================================================
# 1) Pydantic schemas (STRICT JSON outputs from the LLMs)
# =============================================================================

class NumericEvidence(BaseModel):
    value: Optional[confloat(ge=0, allow_inf_nan=False)] = Field(None, description="Parsed numeric value if available")
    unit: Optional[str] = None  # e.g., 'x', '%', 'USD'
    quote: Optional[str] = None
    page: Optional[int] = None       # 1-based
    bbox: Optional[List[int]] = None # [x1,y1,x2,y2] if available
    confidence: confloat(ge=0, le=1) = 0.0

class CovenantItem(BaseModel):
    item: str
    threshold: Optional[str] = None
    observed: Optional[str] = None
    status: str  # 'compliant' | 'breached' | 'unclear'
    evidence: Optional[NumericEvidence] = None

class CovenantExtraction(BaseModel):
    # List of extracted covenants; empty list by default
    covenants: List[CovenantItem] = Field(default_factory=list, min_length=0)
    # Overall extraction confidence in [0, 1]; disallow NaN/Inf explicitly
    confidence: confloat(ge=0, le=1, allow_inf_nan=False) = 0.0


class Discrepancy(BaseModel):
    feature: str
    computed_value: Optional[str] = None
    stated_value: Optional[str] = None
    issue: str  # e.g., 'mismatch', 'missing_in_text', 'missing_in_sheet'
    evidence: Optional[NumericEvidence] = None
    severity: str = "medium"  # low|medium|high

class DiscrepancyReport(BaseModel):
    discrepancies: List[Discrepancy] = Field(default_factory=list, min_length=0)
    confidence: confloat(ge=0, le=1, allow_inf_nan=False) = 0.0



# =============================================================================
# 2) JSON-safe text generation helper for Hugging Face LLMs
# =============================================================================

@dataclass
class HFJsonClientConfig:
    model_path: str                      # local dir or HF repo id
    device: str = "auto"
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05


class HFJsonClient:
    """
    Simple wrapper that:
      - Builds a strict JSON extraction prompt for a *text-only* LLM
      - Calls HF transformers pipeline
      - Tries to parse JSON and auto-recovers from common formatting mistakes
    """
    def __init__(self, cfg: HFJsonClientConfig):
        logger.info(f"Loading text LLM: {cfg.model_path}")
        self.pipe = pipeline(
            "text-generation",
            model=cfg.model_path,
            tokenizer=cfg.model_path,
            device_map=cfg.device,
            model_kwargs={"torch_dtype": "auto"},
        )
        self.cfg = cfg

    def _gen(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0.0,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return out[len(prompt):] if out.startswith(prompt) else out

    @staticmethod
    def _strip_to_json(text: str) -> str:
        """
        Extract first {...} or [...] block; removes markdown fences.
        """
        txt = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
        # Try object
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m: return m.group(0)
        # Try array fallback
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        return m.group(0) if m else txt

    def generate_json(self, system: str, instruction: str, schema_hint: str, retries: int = 2) -> Dict[str, Any]:
        """
        Ask for STRICT JSON. On parse failure, retry with explicit correction.
        """
        # Minimal "chat" style prompt for generic instruct models
        prompt = (
            f"<<SYS>>\n{system}\n<</SYS>>\n"
            f"[TASK]\n{instruction}\n\n"
            "Respond with STRICT JSON only. Do not include any extra text."
            f"\n\n[SCHEMA]\n{schema_hint}\n"
        )

        last_err = None
        for step in range(retries + 1):
            raw = self._gen(prompt)
            candidate = self._strip_to_json(raw)
            try:
                return json.loads(candidate)
            except Exception as e:
                last_err = e
                # Retry with a corrective reminder
                prompt += "\n\n[REMINDER] Output must be valid JSON only. No prose, no code fences."
                time.sleep(0.5)
        raise ValueError(f"Model did not return valid JSON after {retries+1} attempts. Last error: {last_err}")


# =============================================================================
# 3) Vision-language extraction using Qwen2(-.5)-VL (per-page extraction)
# =============================================================================

class QwenVLMExtractor:
    """
    Extracts small, well-defined facts from page images and returns JSON with value/unit/quote/page/confidence.
    Requires qwen-vl-utils and a Qwen2-VL or Qwen2.5-VL checkpoint.
    """
    def __init__(self, vl_model: str = "Qwen/Qwen2.5-VL-2B-Instruct", device: str = "auto"):
        if not HAS_QWEN:
            raise RuntimeError("Qwen VLM not available. Install qwen-vl-utils & compatible transformers.")
        logger.info(f"Loading VLM: {vl_model}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(vl_model, torch_dtype="auto", device_map=device)
        self.proc  = AutoProcessor.from_pretrained(vl_model)

    def extract_numeric_from_pages(
        self,
        page_images: List[Image.Image],
        question: str,
        units_hint: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> List[NumericEvidence]:
        """
        Ask the VLM to find the numeric answer + quote on each page separately.
        """
        results: List[NumericEvidence] = []
        for i, im in enumerate(page_images, start=1):
            msg = [{
                "role":"user",
                "content":[
                    {"type":"image", "image": im},
                    {"type":"text",  "text": (
                        "Task: Extract the requested numeric value from this page.\n"
                        f"Question: {question}\n"
                        f"Units hint: {units_hint or 'n/a'}\n"
                        "Return STRICT JSON with keys: {value: number|null, unit: string|null, quote: string, confidence: 0..1}\n"
                        "Only return JSON; no extra text."
                    )}
                ]
            }]

            templated = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(msg)
            inputs = self.proc(text=[templated], images=images, videos=videos, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
            text = self.proc.batch_decode(gen, skip_special_tokens=True)[0]

            # Try to parse JSON-ish
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            payload = {"value": None, "unit": None, "quote": None, "confidence": 0.0}
            if m:
                try:
                    payload = json.loads(m.group(0))
                except Exception:
                    pass

            results.append(NumericEvidence(
                value=payload.get("value"),
                unit=payload.get("unit"),
                quote=payload.get("quote"),
                page=i,
                confidence=float(payload.get("confidence", 0.0))
            ))
        return results


# =============================================================================
# 4) Task-specific AI helpers (surgical prompts)
# =============================================================================

def _schema_hint_numeric() -> str:
    return json.dumps(NumericEvidence.model_json_schema(), indent=2)

def _schema_hint_covenant() -> str:
    return json.dumps(CovenantExtraction.model_json_schema(), indent=2)

def _schema_hint_discrepancy() -> str:
    return json.dumps(DiscrepancyReport.model_json_schema(), indent=2)


def extract_interest_coverage_from_text(
    client: HFJsonClient,
    doc_text: str,
    unit_hint: str = "x",
    pages: Optional[List[str]] = None,
) -> NumericEvidence:
    """
    Finds interest coverage in narrative text; returns NumericEvidence with page back-link if pages provided.
    """
    system = (
        "You are a meticulous financial analyst. Extract a single numeric value for the requested metric "
        "from the provided document text and return STRICT JSON."
    )
    instruction = (
        "Metric: Interest coverage ratio.\n"
        "Return JSON with value (number or null), unit (likely 'x'), an exact quote with the number, "
        "and a confidence (0..1). Prefer the most recent/current period if multiple are present.\n"
        "Document text:\n" + doc_text[:18000]  # guardrail
    )
    data = client.generate_json(system, instruction, _schema_hint_numeric())
    try:
        result = NumericEvidence(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid LLM JSON for interest coverage: {e}")

    # Optional: back-map quote to a page index by searching page texts
    if result.quote and pages:
        for idx, page_txt in enumerate(pages, start=1):
            if result.quote[:80] in page_txt:  # fuzzy could be used here
                result.page = idx
                break
    if result.unit is None:
        result.unit = unit_hint
    return result


def extract_covenants_and_breaches(
    client: HFJsonClient,
    doc_text: str,
) -> CovenantExtraction:
    """
    Extract covenants and mark status as 'compliant' / 'breached' / 'unclear'.
    """
    system = (
        "You extract financial covenants from credit documents and assess if the text indicates compliance or breach. "
        "Always produce STRICT JSON."
    )
    instruction = (
        "From the document text, list all financial covenants and whether the narrative indicates compliance or breach. "
        "If insufficient evidence exists, use 'unclear'. If a numeric threshold is stated, include it. "
        "If an observed numeric is stated, include it too. Include a short quote for each (in evidence.quote) "
        "and a confidence 0..1 for the overall extraction.\n"
        "Document text:\n" + doc_text[:18000]
    )
    data = client.generate_json(system, instruction, _schema_hint_covenant())
    try:
        result = CovenantExtraction(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid LLM JSON for covenants: {e}")
    return result


def run_discrepancy_check(
    client: HFJsonClient,
    features: Dict[str, Any],
    doc_text: str,
    critical: Optional[List[str]] = None
) -> DiscrepancyReport:
    """
    Compares deterministic features (e.g., from Excel) vs. narrative claims in text.
    Ask LLM to identify mismatches and missing claims; return JSON.
    """
    critical = critical or []
    system = (
        "You are a validator that compares quantitative features with a document's narrative. "
        "Identify mismatches or missing mentions and return STRICT JSON."
    )
    instruction = (
        "Given the FEATURES and the DOCUMENT TEXT, find any discrepancies:\n"
        "- If a feature is present in FEATURES but a different value is stated in the text → issue='mismatch'\n"
        "- If a critical feature is missing from the text → issue='missing_in_text'\n"
        "- If the text claims a value but FEATURES lacks it → issue='missing_in_sheet'\n"
        "Include evidence quotes where possible and suggest severity ('high' for critical items).\n\n"
        f"CRITICAL FEATURES: {critical}\n"
        f"FEATURES (JSON): {json.dumps(features)[:8000]}\n\n"
        "DOCUMENT TEXT:\n" + doc_text[:14000]
    )
    data = client.generate_json(system, instruction, _schema_hint_discrepancy())
    try:
        result = DiscrepancyReport(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid LLM JSON for discrepancy report: {e}")
    return result


# =============================================================================
# 5) Confidence gating utility
# =============================================================================

def gate_for_review(*confidences: float, min_conf: float = 0.65) -> bool:
    """
    Returns True if any confidence is below threshold → route to human REVIEW.
    """
    return any((c is None) or (c < min_conf) for c in confidences)


# =============================================================================
# 6) Example orchestration
# =============================================================================

def example_usage():
    """
    Example showing:
      - text LLM for JSON extraction (local 'AdaptLLM/finance-LLM' or your finetune)
      - VLM extraction stub (if you have page images)
      - confidence gating and discrepancy checks
    """
    # --- 6.1 Text LLM client (use local snapshot path for enterprise) ---
    llm_cfg = HFJsonClientConfig(
        model_path="AdaptLLM/finance-LLM",  # or r"C:\models\finance-LLM"
        temperature=0.0
    )
    client = HFJsonClient(llm_cfg)

    # --- 6.2 Assume you already normalized your document into text + per-page text ---
    doc_text = "The company reports an interest coverage ratio of 2.3x for the fiscal year 2025. "\
               "Net leverage must remain below 3.5x; management states it is 3.2x as of Q4. "\
               "DSCR is 1.28x which satisfies the minimum of 1.25x."
    page_texts = [
        "Page 1 ... interest coverage ratio of 2.3x for FY2025 ...",
        "Page 2 ... DSCR is 1.28x ... minimum 1.25x ...",
        "Page 3 ... Net leverage must remain below 3.5x; stated 3.2x ..."
    ]

    # --- 6.3 Extract interest coverage from text ---
    ic = extract_interest_coverage_from_text(client, doc_text, unit_hint="x", pages=page_texts)
    print("Interest coverage:", ic.model_dump())

    # --- 6.4 Extract covenants & breaches from text ---
    cov = extract_covenants_and_breaches(client, doc_text)
    print("Covenants:", cov.model_dump())

    # --- 6.5 Compare with deterministic features (e.g., computed from Excel) ---
    features = {
        "interest_coverage": 2.30,
        "dscr": 1.24,                 # <-- intentionally lower to trigger a mismatch vs stated 1.28x
        "net_leverage": 3.20
    }
    disc = run_discrepancy_check(client, features, doc_text, critical=["dscr"])
    print("Discrepancies:", disc.model_dump())

    # --- 6.6 Confidence gating ---
    send_to_review = gate_for_review(ic.confidence, cov.confidence, disc.confidence, min_conf=0.7)
    print("Route to human REVIEW?", send_to_review)

    # --- 6.7 VLM per-page numeric extraction (optional) ---
    # If you have page images (PIL.Image), you can extract on each page directly.
    # Example (disabled here because we didn't render any real images):
    if False and HAS_QWEN:
        qwen = QwenVLMExtractor("Qwen/Qwen2.5-VL-2B-Instruct")
        # page_images = [Image.open("page1.png"), Image.open("page2.png")]
        # vlm_vals = qwen.extract_numeric_from_pages(page_images, question="What is the DSCR?", units_hint="x")
        # print([v.model_dump() for v in vlm_vals])





# --- Offline, regex-based extractors (no internet, no HF models) ---
import re
from typing import Dict, Any, List

def _find_snippet(text: str, start: int, end: int, pad: int = 60) -> str:
    s = max(0, start - pad); e = min(len(text), end + pad)
    return text[s:e].strip()

def offline_extract_interest_coverage(doc_text: str, pages: List[str] | None = None) -> NumericEvidence:
    """
    Extracts 'interest coverage' like "... interest coverage ... 2.6x ..."
    Returns NumericEvidence; page is resolved approximately by searching page texts.
    """
    m = re.search(r"interest\s+coverage[^0-9]{0,40}(\d+(?:\.\d+)?)\s*x", doc_text, flags=re.I)
    if not m:
        return NumericEvidence(value=None, unit="x", quote=None, page=None, confidence=0.0)
    val = float(m.group(1))
    quote = _find_snippet(doc_text, m.start(), m.end())
    page_no = None
    if quote and pages:
        # rough page back-link
        key = quote[:80]
        for i, ptxt in enumerate(pages, start=1):
            if key and key in ptxt:
                page_no = i
                break
    return NumericEvidence(value=val, unit="x", quote=quote, page=page_no, confidence=0.95)

def offline_extract_covenants(doc_text: str) -> CovenantExtraction:
    """
    Looks for typical bullet lines like:
    'Minimum Interest Coverage: threshold 2.0x; reported 2.6x — compliant.'
    'Minimum DSCR: threshold 1.25x; reported 1.10x — breached.'
    'Maximum Net Leverage: threshold 3.5x; reported 3.8x — breached.'
    """
    covenants: List[CovenantItem] = []
    patterns = [
        ("Minimum Interest Coverage", r"Minimum\s+Interest\s+Coverage:\s*threshold\s*([0-9.]+)x;?\s*reported\s*([0-9.]+)x\s*—\s*([A-Za-z]+)"),
        ("Minimum DSCR",              r"Minimum\s+DSCR:\s*threshold\s*([0-9.]+)x;?\s*reported\s*([0-9.]+)x\s*—\s*([A-Za-z]+)"),
        ("Maximum Net Leverage",      r"Maximum\s+Net\s+Leverage:\s*threshold\s*([0-9.]+)x;?\s*reported\s*([0-9.]+)x\s*—\s*([A-Za-z]+)"),
    ]
    for title, pat in patterns:
        for m in re.finditer(pat, doc_text, flags=re.I):
            thr = m.group(1)
            obs = m.group(2)
            status_raw = m.group(3).lower().strip(".")
            status = "compliant" if "compliant" in status_raw else ("breached" if "breach" in status_raw or "breached" in status_raw else "unclear")
            quote = _find_snippet(doc_text, m.start(), m.end())
            covenants.append(CovenantItem(
                item=title,
                threshold=f"{thr}x",
                observed=f"{obs}x",
                status=status,
                evidence=NumericEvidence(value=float(obs), unit="x", quote=quote, page=None, confidence=0.95)
            ))
    # If nothing matched, try a weaker signal (words 'compliant' / 'not in compliance')
    if not covenants:
        txt = doc_text.lower()
        if "remains in compliance with all financial covenants" in txt:
            covenants.append(CovenantItem(item="All covenants", status="compliant"))
        if "not in compliance" in txt or "breach" in txt:
            covenants.append(CovenantItem(item="One or more covenants", status="breached"))
    return CovenantExtraction(covenants=covenants, confidence=0.9 if covenants else 0.0)

def offline_discrepancy_check(features: Dict[str, Any], doc_text: str, critical: List[str] | None = None) -> DiscrepancyReport:
    """
    Simple, local discrepancy finder:
    - If a numeric feature exists and the same number isn't roughly visible in text, flag 'missing_in_text'
    - If text states a number for IC/DSCR and it differs by > 0.05, flag 'mismatch'
    """
    import math
    discrepancies: List[Discrepancy] = []
    critical = set(critical or [])

    # helper to find a number near a metric keyword
    def find_num_near(keyword: str) -> float | None:
        m = re.search(fr"{keyword}[^0-9]{{0,40}}(\d+(?:\.\d+)?)\s*x", doc_text, flags=re.I)
        return float(m.group(1)) if m else None

    # interest coverage
    ic_text = find_num_near("interest\\s+coverage")
    if features.get("interest_coverage") is not None and ic_text is not None:
        if abs(features["interest_coverage"] - ic_text) > 0.05:
            discrepancies.append(Discrepancy(
                feature="interest_coverage",
                computed_value=str(features["interest_coverage"]),
                stated_value=str(ic_text),
                issue="mismatch",
                evidence=NumericEvidence(value=ic_text, unit="x", quote=None, page=None, confidence=0.6),
                severity="medium"
            ))
    elif features.get("interest_coverage") is not None and ic_text is None:
        discrepancies.append(Discrepancy(
            feature="interest_coverage",
            computed_value=str(features["interest_coverage"]),
            stated_value=None,
            issue="missing_in_text",
            evidence=None,
            severity="low"
        ))

    # dscr
    dscr_text = find_num_near("dscr")
    if features.get("dscr") is not None and dscr_text is not None:
        if abs(features["dscr"] - dscr_text) > 0.05:
            discrepancies.append(Discrepancy(
                feature="dscr",
                computed_value=str(features["dscr"]),
                stated_value=str(dscr_text),
                issue="mismatch",
                evidence=NumericEvidence(value=dscr_text, unit="x", quote=None, page=None, confidence=0.6),
                severity="high" if "dscr" in critical else "medium"
            ))
    elif features.get("dscr") is not None and dscr_text is None:
        discrepancies.append(Discrepancy(
            feature="dscr",
            computed_value=str(features["dscr"]),
            stated_value=None,
            issue="missing_in_text",
            evidence=None,
            severity="high" if "dscr" in critical else "low"
        ))

    # net leverage (optional: compare presence)
    nl_text = find_num_near("net\\s+leverage")
    if features.get("net_leverage") is not None and nl_text is not None:
        if abs(features["net_leverage"] - nl_text) > 0.05:
            discrepancies.append(Discrepancy(
                feature="net_leverage",
                computed_value=str(features["net_leverage"]),
                stated_value=str(nl_text),
                issue="mismatch",
                evidence=NumericEvidence(value=nl_text, unit="x", quote=None, page=None, confidence=0.6),
                severity="medium"
            ))

    return DiscrepancyReport(discrepancies=discrepancies, confidence=0.8)

if __name__ == "__main__":
    example_usage()
# ai_assist_risk_endpoint.py
"""
Extended version of ai_assist_risk.py with SageMaker endpoint support.
This replaces the local HFJsonClient with a SageMaker endpoint client.
"""
from __future__ import annotations
import json, re, time, logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import boto3

from pydantic import BaseModel, Field, confloat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import all the Pydantic schemas from the original module
from ai_assist_risk import (
    NumericEvidence,
    CovenantItem,
    CovenantExtraction,
    Discrepancy,
    DiscrepancyReport,
    # Import offline functions
    offline_extract_interest_coverage,
    offline_extract_covenants,
    offline_discrepancy_check,
    gate_for_review,
)


# =============================================================================
# SageMaker Endpoint Client (replaces HFJsonClient)
# =============================================================================

@dataclass
class SageMakerEndpointConfig:
    """Configuration for SageMaker endpoint."""
    endpoint_name: str
    region_name: Optional[str] = None
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0


class SageMakerEndpointClient:
    """
    Client that calls a SageMaker endpoint for JSON extraction.
    This is a drop-in replacement for HFJsonClient that uses a deployed endpoint.
    """
    
    def __init__(self, cfg: SageMakerEndpointConfig):
        logger.info(f"Initializing SageMaker endpoint client: {cfg.endpoint_name}")
        self.cfg = cfg
        self.runtime_client = boto3.client(
            'sagemaker-runtime',
            region_name=cfg.region_name
        )
        
        # Verify endpoint exists and is in service
        sagemaker_client = boto3.client('sagemaker', region_name=cfg.region_name)
        try:
            response = sagemaker_client.describe_endpoint(
                EndpointName=cfg.endpoint_name
            )
            status = response['EndpointStatus']
            if status != 'InService':
                raise RuntimeError(
                    f"Endpoint {cfg.endpoint_name} is not in service (status: {status})"
                )
            logger.info(f"Endpoint {cfg.endpoint_name} is ready (status: {status})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify endpoint {cfg.endpoint_name}: {e}"
            )
    
    def _invoke_endpoint(self, prompt: str) -> str:
        """Invoke the SageMaker endpoint with a prompt."""

        payload = {
            "inputs": prompt, # Use the truncated version here
            "parameters": {
                "max_new_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                # "return_full_text": False,
                "do_sample": self.cfg.temperature > 0.0,
            }
        }
        
        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.cfg.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Format: [{"generated_text": "..."}]
                return result[0].get('generated_text', '')
            elif isinstance(result, dict):
                # Format: {"generated_text": "..."}
                return result.get('generated_text', '')
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Endpoint invocation failed: {e}")
            raise
    
    @staticmethod
    def _strip_to_json(text: str) -> str:
        """Extract first {...} or [...] block; removes markdown fences."""
        txt = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
        # Try object
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            return m.group(0)
        # Try array fallback
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        return m.group(0) if m else txt
    
    def generate_json(
        self,
        system: str,
        instruction: str,
        schema_hint: str,
        retries: int = 2
    ) -> Dict[str, Any]:
        """
        Ask for STRICT JSON via the endpoint. On parse failure, retry.
        This matches the signature of HFJsonClient.generate_json().
        """
        # Build prompt (matching original format)
        prompt = (
            f"<<SYS>>\n{system}\n<</SYS>>\n"
            f"[TASK]\n{instruction}\n\n"
            "Respond with STRICT JSON only. Do not include any extra text."
            f"\n\n[SCHEMA]\n{schema_hint}\n"
        )
        
        last_err = None
        for step in range(retries + 1):
            raw = self._invoke_endpoint(prompt)
            candidate = self._strip_to_json(raw)
            try:
                return json.loads(candidate)
            except Exception as e:
                last_err = e
                # Retry with corrective reminder
                prompt += "\n\n[REMINDER] Output must be valid JSON only. No prose, no code fences."
                time.sleep(0.5)
        
        raise ValueError(
            f"Endpoint did not return valid JSON after {retries+1} attempts. "
            f"Last error: {last_err}"
        )


# =============================================================================
# Extraction functions using SageMaker endpoint
# =============================================================================

def extract_interest_coverage_from_text_endpoint(
    client: SageMakerEndpointClient,
    doc_text: str,
    unit_hint: str = "x",
    pages: Optional[List[str]] = None
) -> NumericEvidence:
    """
    Extract interest coverage using SageMaker endpoint.
    Drop-in replacement for extract_interest_coverage_from_text().
    """
    system = (
        "You are a financial analyst assistant that extracts numeric facts "
        "from credit documents. You return JSON only."
    )
    
    instruction = (
        f"Extract the interest coverage ratio from this text.\n\n"
        f"TEXT:\n{doc_text[:12000]}\n\n"
        "Return the value, unit, and a short verbatim quote from the text."
    )
    
    schema_hint = """{
  "value": <number or null>,
  "unit": "x",
  "quote": "<exact quote>",
  "page": <number or null>,
  "confidence": <0.0 to 1.0>
}"""
    
    try:
        result = client.generate_json(system, instruction, schema_hint)
        return NumericEvidence(**result)
    except Exception as e:
        logger.warning(f"Interest coverage extraction failed: {e}")
        return NumericEvidence(value=None, unit=unit_hint, confidence=0.0)


def extract_covenants_and_breaches_endpoint(
    client: SageMakerEndpointClient,
    doc_text: str
) -> CovenantExtraction:
    """
    Extract covenants using SageMaker endpoint.
    Drop-in replacement for extract_covenants_and_breaches().
    """
    system = (
        "You are a credit analyst. Extract all financial covenants from this document. "
        "For each covenant, identify threshold, observed value, and whether it's breached. "
        "Return JSON only."
    )
    
    instruction = (
        f"TEXT:\n{doc_text[:12000]}\n\n"
        "List all covenants with their status (compliant/breached/unclear)."
    )
    
    schema_hint = """{
  "covenants": [
    {
      "item": "<covenant name>",
      "threshold": "<threshold value>",
      "observed": "<observed value>",
      "status": "compliant|breached|unclear",
      "evidence": {"value": <number>, "unit": "x", "quote": "...", "confidence": 0.9}
    }
  ],
  "confidence": <0.0 to 1.0>
}"""
    
    try:
        result = client.generate_json(system, instruction, schema_hint)
        return CovenantExtraction(**result)
    except Exception as e:
        logger.warning(f"Covenant extraction failed: {e}")
        return CovenantExtraction(covenants=[], confidence=0.0)


def run_discrepancy_check_endpoint(
    client: SageMakerEndpointClient,
    features: Dict[str, Any],
    doc_text: str,
    critical: Optional[List[str]] = None
) -> DiscrepancyReport:
    """
    Check for discrepancies using SageMaker endpoint.
    Drop-in replacement for run_discrepancy_check().
    """
    system = (
        "You are a data quality analyst. Compare computed metrics against "
        "what's stated in the document. Flag any mismatches."
    )
    
    instruction = (
        f"COMPUTED FEATURES:\n{json.dumps(features, indent=2)}\n\n"
        f"DOCUMENT TEXT:\n{doc_text[:12000]}\n\n"
        "Identify any discrepancies between computed values and stated values."
    )
    
    schema_hint = """{
  "discrepancies": [
    {
      "feature": "<feature name>",
      "computed_value": "<computed>",
      "stated_value": "<from text>",
      "issue": "mismatch|missing_in_text|missing_in_sheet",
      "severity": "low|medium|high"
    }
  ],
  "confidence": <0.0 to 1.0>
}"""
    
    try:
        result = client.generate_json(system, instruction, schema_hint)
        return DiscrepancyReport(**result)
    except Exception as e:
        logger.warning(f"Discrepancy check failed: {e}")
        return DiscrepancyReport(discrepancies=[], confidence=0.0)


# =============================================================================
# Example usage
# =============================================================================

def example_endpoint_usage():
    """Example showing how to use the SageMaker endpoint client."""
    
    # Configure endpoint client
    endpoint_cfg = SageMakerEndpointConfig(
        endpoint_name="finance-llm-endpoint",  # Your deployed endpoint
        region_name="us-east-1",
        temperature=0.0
    )
    
    client = SageMakerEndpointClient(endpoint_cfg)
    
    # Sample document text
    doc_text = """
    The company reports an interest coverage ratio of 2.3x for fiscal year 2025.
    Net leverage must remain below 3.5x; management states it is 3.2x as of Q4.
    DSCR is 1.28x which satisfies the minimum covenant of 1.25x.
    """
    
    # Extract interest coverage
    ic = extract_interest_coverage_from_text_endpoint(
        client, doc_text, unit_hint="x"
    )
    print("Interest coverage:", ic.model_dump())
    
    # Extract covenants
    cov = extract_covenants_and_breaches_endpoint(client, doc_text)
    print("Covenants:", cov.model_dump())
    
    # Check discrepancies
    features = {
        "interest_coverage": 2.30,
        "dscr": 1.24,  # Intentionally different to trigger mismatch
        "net_leverage": 3.20
    }
    disc = run_discrepancy_check_endpoint(client, features, doc_text, critical=["dscr"])
    print("Discrepancies:", disc.model_dump())
    
    # Confidence gating
    send_to_review = gate_for_review(
        ic.confidence, cov.confidence, disc.confidence, min_conf=0.7
    )
    print("Route to REVIEW?", send_to_review)


if __name__ == "__main__":
    example_endpoint_usage()
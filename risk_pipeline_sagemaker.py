# risk_pipeline_sagemaker.py
"""
AWS SageMaker compatible credit risk pipeline with S3 integration.
Reads PDFs and Excel files from S3, processes them, and writes outputs back to S3.
"""
from __future__ import annotations
import os, sys, json, argparse, tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import yaml
import pandas as pd
from PyPDF2 import PdfReader
from jinja2 import Template
import boto3
from botocore.exceptions import ClientError

from ai_assist_risk import (
    HFJsonClientConfig, HFJsonClient,
    extract_interest_coverage_from_text,
    extract_covenants_and_breaches,
    run_discrepancy_check,
    gate_for_review,
    NumericEvidence, CovenantExtraction, DiscrepancyReport,
    offline_extract_interest_coverage,
    offline_extract_covenants,
    offline_discrepancy_check,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Credit Risk Decision</title>
<style>
 body{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:24px}
 .label{font-size:22px;font-weight:600}
 .pill{display:inline-block;padding:4px 10px;border-radius:12px;color:#fff}
 .HEALTHY{background:#14866d}.REVIEW{background:#956400}.UNHEALTHY{background:#b3261e}
 .rule{margin:10px 0;padding:10px;border:1px solid #eee;border-radius:8px}
 .pass{color:#14866d}.fail{color:#b3261e}
 pre{white-space:pre-wrap}
</style></head><body>
<div class="label">Decision: <span class="pill {{label}}">{{label}}</span></div>
<div>Score: {{score}} (threshold: {{healthy_threshold}})</div>
<hr/>
<h3>Rule Outcomes</h3>
{% for r in outcomes %}
  <div class="rule">
    <div><b>{{r.id}} — {{r.name}}</b></div>
    <div class="{{ 'pass' if r.passed else 'fail' }}">{{r.message}}</div>
    <div>Weight: {{r.weight}} {% if r.hard_fail %} | <b>Hard-fail</b>{% endif %}</div>
  </div>
{% endfor %}
<hr/>
<h3>AI Evidence</h3>
<h4>Interest coverage</h4>
<pre>{{ic_json}}</pre>
<h4>Covenants</h4>
<pre>{{cov_json}}</pre>
<h4>Discrepancy report</h4>
<pre>{{disc_json}}</pre>
</body></html>
"""


class S3Handler:
    """Handles all S3 operations for reading and writing files."""
    
    def __init__(self, bucket_name: str, region_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        logger.info(f"Initialized S3 handler for bucket: {bucket_name}")
    
    def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download a file from S3 to local path."""
        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Successfully downloaded {s3_key}")
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise
    
    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """Upload a local file to S3."""
        try:
            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded to {s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload to {s3_key}: {e}")
            raise
    
    def upload_string(self, content: str, s3_key: str, content_type: str = 'text/plain') -> None:
        """Upload string content directly to S3."""
        try:
            logger.info(f"Uploading content to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content.encode('utf-8'),
                ContentType=content_type
            )
            logger.info(f"Successfully uploaded content to {s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload content to {s3_key}: {e}")
            raise
    
    def list_files(self, prefix: str = '') -> List[str]:
        """List all files in the bucket with given prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            raise
    
    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False


class AttrDict(dict):
    """Dictionary with attribute access."""
    def __getattr__(self, k):
        v = self.get(k)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            return AttrDict(v)
        return v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def safe_eval(expr: str, env: Dict[str, Any]) -> bool:
    """Safely evaluate a boolean expression."""
    return bool(eval(expr, {"__builtins__": {}}, env))


def render_text(tmpl: str, env: Dict[str, Any]) -> str:
    """Render a Jinja2 template with given environment."""
    return Template(tmpl).render(**env)


def read_policy(path: Path) -> Dict[str, Any]:
    """Read YAML policy file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pdf_texts(pdf_path: Path) -> List[str]:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def load_excel_metrics(excel_path: Path) -> Dict[str, float]:
    """Load metrics from Excel file with 'Metrics' sheet containing Metric and Value columns."""
    df = pd.read_excel(excel_path, sheet_name='Metrics', engine='openpyxl')
    df = df.dropna()
    metrics = {}
    for _, row in df.iterrows():
        key = str(row['Metric']).strip()
        try:
            val = float(row['Value'])
        except Exception:
            continue
        metrics[key] = val
    return metrics


def evaluate_rules(policy: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate policy rules against extracted features."""
    thresholds = policy.get("thresholds", {})
    rules_def = policy.get("rules", [])
    aggreg = policy.get("aggregation", {})
    labels = aggreg.get("labels", {"healthy":"HEALTHY", "review":"REVIEW", "unhealthy":"UNHEALTHY"})

    env = {"features": AttrDict(features), "thresholds": AttrDict(thresholds)}

    outcomes = []
    score = 0.0
    weight_sum = 0.0
    any_hard_fail = False

    for rd in rules_def:
        rid = rd['id']
        name = rd.get('name', rid)
        when = rd.get('when')
        cond = rd.get('condition', 'False')
        msg_pass = rd.get('pass', 'Pass')
        msg_fail = rd.get('fail', 'Fail')
        weight = float(rd.get('weight', 0.0))
        hard_fail = bool(rd.get('hard_fail', False))

        should_eval = True
        if when:
            should_eval = safe_eval(when, env)
        passed = safe_eval(cond, env) if should_eval else True

        message = Template(msg_pass if passed else msg_fail).render(**env)
        outcomes.append({
            'id': rid, 'name': name, 'passed': passed, 'message': message,
            'weight': weight, 'hard_fail': hard_fail
        })

        if passed:
            score += weight
        weight_sum += weight
        if hard_fail and not passed:
            any_hard_fail = True

    norm_score = (score / weight_sum) if weight_sum > 0 else 1.0
    healthy_threshold = float(aggreg.get('healthy_threshold', 0.70))

    if any_hard_fail:
        label = labels['unhealthy']
    else:
        if norm_score >= healthy_threshold:
            label = labels['healthy']
        elif norm_score >= (healthy_threshold * 0.85):
            label = labels['review']
        else:
            label = labels['unhealthy']

    return {
        'label': label,
        'score': round(norm_score, 3),
        'outcomes': outcomes,
        'healthy_threshold': healthy_threshold,
        'hard_fail_triggered': any_hard_fail
    }


def process_risk_assessment(
    s3_handler: S3Handler,
    pdf_s3_key: str,
    excel_s3_key: Optional[str],
    policy_s3_key: str,
    output_prefix: str,
    text_llm: str,
    offline: bool = False
) -> Dict[str, Any]:
    """
    Main processing function that:
    1. Downloads files from S3
    2. Processes them
    3. Uploads results back to S3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Download files from S3
        logger.info("Downloading files from S3...")
        pdf_local = tmp_path / "input.pdf"
        policy_local = tmp_path / "policy.yaml"
        
        s3_handler.download_file(pdf_s3_key, pdf_local)
        s3_handler.download_file(policy_s3_key, policy_local)
        
        excel_local = None
        if excel_s3_key:
            excel_local = tmp_path / "metrics.xlsx"
            s3_handler.download_file(excel_s3_key, excel_local)
        
        # Load policy and extract text from PDF
        logger.info("Loading policy and extracting PDF text...")
        policy = read_policy(policy_local)
        page_texts = load_pdf_texts(pdf_local)
        doc_text = "\n\n".join(page_texts)[:180000]
        
        # Process based on offline mode
        if offline:
            logger.info("Running in OFFLINE mode (regex-based extraction)")
            ic = offline_extract_interest_coverage(doc_text, pages=page_texts)
            cov = offline_extract_covenants(doc_text)
            
            features: Dict[str, Any] = {
                "interest_coverage": ic.value,
                "dscr": None,
                "net_leverage": None,
                "covenant_breach_count": len([c for c in cov.covenants if c.status == "breached"]),
            }
            
            if excel_local:
                metrics = load_excel_metrics(excel_local)
                features["dscr"] = metrics.get("DSCR", features["dscr"])
                features["net_leverage"] = metrics.get("NetLeverage", features["net_leverage"])
            
            disc = offline_discrepancy_check(features, doc_text, critical=["dscr"])
            
        else:
            logger.info("Running in ONLINE mode (LLM-based extraction)")
            llm_cfg = HFJsonClientConfig(model_path=text_llm, temperature=0.0)
            client = HFJsonClient(llm_cfg)
            
            ic = extract_interest_coverage_from_text(client, doc_text, unit_hint='x', pages=page_texts)
            cov = extract_covenants_and_breaches(client, doc_text)
            
            features: Dict[str, Any] = {
                "interest_coverage": ic.value,
                "dscr": None,
                "net_leverage": None,
                "covenant_breach_count": len([c for c in cov.covenants if c.status == "breached"]),
            }
            
            if excel_local:
                metrics = load_excel_metrics(excel_local)
                features["dscr"] = metrics.get("DSCR", features["dscr"])
                features["net_leverage"] = metrics.get("NetLeverage", features["net_leverage"])
            
            disc = run_discrepancy_check(client, features, doc_text, critical=["dscr"])
        
        # Evaluate rules
        logger.info("Evaluating policy rules...")
        decision = evaluate_rules(policy, features)
        
        # Apply confidence gating
        needs_review = gate_for_review(ic.confidence, cov.confidence, disc.confidence, min_conf=0.70)
        if needs_review and decision['label'] == 'HEALTHY':
            decision['label'] = 'REVIEW'
            logger.info("Low confidence detected - routing to REVIEW")
        
        # Prepare output JSON
        result = {
            'policy_version': policy.get('version', 'na'),
            'pdf_s3_key': pdf_s3_key,
            'excel_s3_key': excel_s3_key,
            'features': features,
            'decision': decision,
            'ai': {
                'interest_coverage': ic.model_dump(),
                'covenants': cov.model_dump(),
                'discrepancies': disc.model_dump(),
            },
        }
        
        # Upload JSON result
        logger.info("Uploading results to S3...")
        json_key = f"{output_prefix}/decision.json"
        s3_handler.upload_string(
            json.dumps(result, indent=2),
            json_key,
            content_type='application/json'
        )
        
        # Generate and upload HTML report
        tpl = Template(HTML_TEMPLATE)
        html = tpl.render(
            label=decision['label'],
            score=decision['score'],
            healthy_threshold=decision['healthy_threshold'],
            outcomes=decision['outcomes'],
            ic_json=json.dumps(ic.model_dump(), indent=2),
            cov_json=json.dumps(cov.model_dump(), indent=2),
            disc_json=json.dumps(disc.model_dump(), indent=2),
        )
        html_key = f"{output_prefix}/decision.html"
        s3_handler.upload_string(html, html_key, content_type='text/html')
        
        logger.info(f"Results uploaded to s3://{s3_handler.bucket_name}/{output_prefix}/")
        
        return {
            'status': 'success',
            'decision_label': decision['label'],
            'decision_score': decision['score'],
            'json_s3_key': json_key,
            'html_s3_key': html_key,
            'result': result
        }


def main():
    """Main entry point for command-line execution."""
    ap = argparse.ArgumentParser(
        description="Credit Risk Agent for AWS SageMaker with S3 integration"
    )
    
    # S3 configuration
    ap.add_argument('--bucket', default='credit-risk-usecase-bucket', 
                    help='S3 bucket name')
    ap.add_argument('--region', default=None, 
                    help='AWS region (optional, uses default if not specified)')
    
    # Input file S3 keys
    ap.add_argument('--pdf-key', required=True, 
                    help='S3 key for PDF file (e.g., input/document.pdf)')
    ap.add_argument('--excel-key', default=None, 
                    help='S3 key for Excel file with Metrics sheet (optional)')
    ap.add_argument('--policy-key', default='config/credit_policy.yaml', 
                    help='S3 key for policy YAML file')
    
    # Output configuration
    ap.add_argument('--output-prefix', default='output', 
                    help='S3 prefix for output files')
    
    # Model configuration
    ap.add_argument('--text-llm', default='AdaptLLM/finance-LLM',
                    help='HuggingFace model path for text LLM')
    ap.add_argument("--offline", action="store_true", 
                    help="Run with local regex-based extractors (no HF downloads)")
    
    args = ap.parse_args()
    
    try:
        # Initialize S3 handler
        s3_handler = S3Handler(args.bucket, args.region)
        
        # Process the risk assessment
        result = process_risk_assessment(
            s3_handler=s3_handler,
            pdf_s3_key=args.pdf_key,
            excel_s3_key=args.excel_key,
            policy_s3_key=args.policy_key,
            output_prefix=args.output_prefix,
            text_llm=args.text_llm,
            offline=args.offline
        )
        
        print(f"\n{'='*60}")
        print(f"Risk Assessment Complete")
        print(f"{'='*60}")
        print(f"Decision: {result['decision_label']}")
        print(f"Score: {result['decision_score']}")
        print(f"\nResults available at:")
        print(f"  JSON: s3://{args.bucket}/{result['json_s3_key']}")
        print(f"  HTML: s3://{args.bucket}/{result['html_s3_key']}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

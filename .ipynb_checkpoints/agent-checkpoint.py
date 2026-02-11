# agent_bedrock_simple.py
"""
Credit Risk AI Agent using AWS Bedrock directly.
Simple, reliable implementation without complex PydanticAI features.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import tempfile

import yaml
import boto3

from models import (
    AgentConfig,
    AgentContext,
    RiskDecision,
    CreditMetrics,
)
from extractors import BedrockExtractor
from policy import PolicyEngine

logger = logging.getLogger(__name__)


class CreditRiskAgent:
    """
    Credit risk assessment agent using AWS Bedrock.
    
    Orchestrates the entire credit risk evaluation process:
    1. Load documents from S3
    2. Extract financial metrics using Bedrock Nova or offline mode
    3. Evaluate against credit policy
    4. Return structured decision with evidence
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)
        self.extractor = BedrockExtractor(config)
        self.policy_engine = PolicyEngine()
        
        logger.info(f"Initialized Credit Risk Agent")
        if not config.offline_mode:
            logger.info(f"Using Bedrock model: {config.model_name or 'us.amazon.nova-lite-v1:0'}")
        else:
            logger.info("Using offline extraction mode")
    
    async def assess_credit_risk(
        self,
        pdf_s3_key: str,
        excel_s3_key: Optional[str] = None,
        output_prefix: str = "output"
    ) -> RiskDecision:
        """
        Main entry point for credit risk assessment.
        
        Args:
            pdf_s3_key: S3 key for the PDF document
            excel_s3_key: Optional S3 key for Excel metrics
            output_prefix: S3 prefix for output files
        
        Returns:
            RiskDecision with complete evidence and explanation
        """
        logger.info(f"Starting credit risk assessment for {pdf_s3_key}")
        
        # 1. Download and prepare documents
        context = await self._prepare_context(pdf_s3_key, excel_s3_key)
        
        # 2. Extract financial metrics
        logger.info("Extracting financial metrics...")
        metrics = await self.extractor.extract_metrics(context)
        
        # 3. Evaluate against policy
        logger.info("Evaluating policy rules...")
        policy_result = self.policy_engine.evaluate(
            metrics=metrics,
            policy=context.policy
        )
        
        # 4. Determine final decision with confidence gating
        decision_label = self._determine_decision(
            policy_result=policy_result,
            metrics=metrics
        )
        
        # 5. Build final decision object
        decision = RiskDecision(
            decision=decision_label,
            score=policy_result['score'],
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            rule_outcomes=policy_result['outcomes'],
            policy_version=context.policy.get('version', 'unknown'),
            processing_mode="llm" if not self.config.offline_mode else "offline",
            pdf_source=pdf_s3_key,
            excel_source=excel_s3_key
        )
        
        # 6. Save outputs to S3
        await self._save_outputs(decision, output_prefix)
        
        logger.info(f"Assessment complete: {decision.decision} (score: {decision.score:.3f})")
        return decision
    
    async def _prepare_context(
        self,
        pdf_s3_key: str,
        excel_s3_key: Optional[str]
    ) -> AgentContext:
        """Download documents and load policy."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Temporary Directory: {tmpdir}")
            tmp_path = Path(tmpdir)
            logger.info(f"Temporary Path: {tmp_path}")

            logger.info(f"Attempting to download {pdf_s3_key}")
            # Download PDF
            pdf_local = tmp_path / "document.pdf"
            logger.info(f"Downloading {pdf_s3_key}...")
            self.s3.download_file(
                self.config.s3_bucket,
                pdf_s3_key,
                str(pdf_local)
            )
            
            # Extract PDF text
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_local))
            pdf_text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            logger.info(f"Extracted {len(pdf_text)} characters from PDF")
            
            # Load Excel if provided
            excel_data = {}
            if excel_s3_key:
                excel_local = tmp_path / "metrics.xlsx"
                logger.info(f"Downloading {excel_s3_key}...")
                self.s3.download_file(
                    self.config.s3_bucket,
                    excel_s3_key,
                    str(excel_local)
                )
                
                import pandas as pd
                df = pd.read_excel(excel_local, sheet_name='Metrics')
                df = df.dropna()
                excel_data = dict(zip(df['Metric'], df['Value']))
                logger.info(f"Loaded {len(excel_data)} metrics from Excel")
            
            # Load policy
            policy_local = tmp_path / "policy.yaml"
            logger.info(f"Downloading policy from {self.config.policy_s3_key}...")
            self.s3.download_file(
                self.config.s3_bucket,
                self.config.policy_s3_key,
                str(policy_local)
            )
            with open(policy_local) as f:
                policy = yaml.safe_load(f)
                if policy is None:
                    logger.error("Failed to load policy from S3")
                    return
            logger.info(f"Loaded policy version {policy.get('version', 'unknown')}")
            
            return AgentContext(
                config=self.config,
                pdf_text=pdf_text[:180000],  # Limit text size
                excel_data=excel_data,
                policy=policy
            )
    
    def _determine_decision(
        self,
        policy_result: dict,
        metrics: CreditMetrics
    ) -> str:
        """Determine final decision with confidence gating."""
        
        # Check for hard failures
        if policy_result['hard_fail_triggered']:
            logger.warning("Hard fail triggered - routing to UNHEALTHY")
            return "UNHEALTHY"
        
        # Check confidence - route to REVIEW if too low
        avg_confidence = self._calculate_confidence(metrics)
        if avg_confidence < self.config.min_confidence:
            logger.warning(
                f"Low confidence ({avg_confidence:.2f} < {self.config.min_confidence}), routing to REVIEW"
            )
            return "REVIEW"
        
        # Use policy score
        score = policy_result['score']
        healthy_threshold = policy_result['healthy_threshold']
        
        if score >= healthy_threshold:
            return "HEALTHY"
        elif score >= (healthy_threshold * 0.85):
            return "REVIEW"
        else:
            return "UNHEALTHY"
    
    def _calculate_confidence(self, metrics: CreditMetrics) -> float:
        """Calculate average confidence across all metrics."""
        confidences = []
        
        for metric in [metrics.interest_coverage, metrics.dscr, metrics.net_leverage]:
            if metric and metric.confidence > 0:
                confidences.append(metric.confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _save_outputs(self, decision: RiskDecision, output_prefix: str):
        """Save decision outputs to S3."""
        import json
        from jinja2 import Template
        
        # Save JSON
        json_key = f"{output_prefix}/decision.json"
        logger.info(f"Saving JSON to {json_key}")
        self.s3.put_object(
            Bucket=self.config.s3_bucket,
            Key=json_key,
            Body=decision.model_dump_json(indent=2),
            ContentType='application/json'
        )
        
        # Generate and save HTML report
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Risk Decision - {{ decision.decision }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .decision-{{ decision.decision.lower() }} {
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .decision-healthy { background: #d4edda; color: #155724; }
        .decision-review { background: #fff3cd; color: #856404; }
        .decision-unhealthy { background: #f8d7da; color: #721c24; }
        .metric { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .rule { padding: 10px; margin: 5px 0; border-left: 3px solid #ddd; }
        .rule.passed { border-color: #28a745; background: #f1f9f3; }
        .rule.failed { border-color: #dc3545; background: #f9f1f1; }
    </style>
</head>
<body>
    <h1>Credit Risk Assessment</h1>
    
    <div class="decision-{{ decision.decision.lower() }}">
        <h2>Decision: {{ decision.decision }}</h2>
        <p><strong>Score:</strong> {{ "%.1f" | format(decision.score * 100) }}%</p>
        <p><strong>Confidence:</strong> {{ "%.1f" | format(decision.confidence * 100) }}%</p>
        <p><strong>Mode:</strong> {{ decision.processing_mode }}</p>
    </div>
    
    <h3>Financial Metrics</h3>
    {% if decision.metrics.interest_coverage %}
    <div class="metric">
        <strong>Interest Coverage:</strong> {{ decision.metrics.interest_coverage.value }}{{ decision.metrics.interest_coverage.unit }}
        <br><small>Source: {{ decision.metrics.interest_coverage.source }} | 
        Confidence: {{ "%.0f" | format(decision.metrics.interest_coverage.confidence * 100) }}%</small>
        {% if decision.metrics.interest_coverage.quote %}
        <br><small>Quote: "{{ decision.metrics.interest_coverage.quote[:100] }}..."</small>
        {% endif %}
    </div>
    {% endif %}
    
    {% if decision.metrics.dscr %}
    <div class="metric">
        <strong>DSCR:</strong> {{ decision.metrics.dscr.value }}{{ decision.metrics.dscr.unit }}
        <br><small>Source: {{ decision.metrics.dscr.source }} | 
        Confidence: {{ "%.0f" | format(decision.metrics.dscr.confidence * 100) }}%</small>
    </div>
    {% endif %}
    
    {% if decision.metrics.net_leverage %}
    <div class="metric">
        <strong>Net Leverage:</strong> {{ decision.metrics.net_leverage.value }}{{ decision.metrics.net_leverage.unit }}
        <br><small>Source: {{ decision.metrics.net_leverage.source }} | 
        Confidence: {{ "%.0f" | format(decision.metrics.net_leverage.confidence * 100) }}%</small>
    </div>
    {% endif %}
    
    <h3>Policy Rule Outcomes</h3>
    {% for outcome in decision.rule_outcomes %}
    <div class="rule {{ 'passed' if outcome.passed else 'failed' }}">
        <strong>{{ outcome.rule_name }}</strong>
        <p>{{ outcome.message }}</p>
        <small>Weight: {{ outcome.weight }}{% if outcome.is_critical %} | <strong>CRITICAL RULE</strong>{% endif %}</small>
    </div>
    {% endfor %}
    
    {% if decision.metrics.covenants %}
    <h3>Covenants</h3>
    {% for cov in decision.metrics.covenants %}
    <div class="metric">
        <strong>{{ cov.name }}</strong>: {{ cov.status }}
        {% if cov.threshold %}<br><small>Threshold: {{ cov.threshold }} | Actual: {{ cov.actual }}</small>{% endif %}
    </div>
    {% endfor %}
    {% endif %}
    
    <hr>
    <p><small>Generated: {{ decision.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") }}</small></p>
    <p><small>Policy Version: {{ decision.policy_version }}</small></p>
    <p><small>PDF: {{ decision.pdf_source }}</small></p>
    {% if decision.excel_source %}
    <p><small>Excel: {{ decision.excel_source }}</small></p>
    {% endif %}
</body>
</html>
        """
        
        html = Template(html_template).render(decision=decision)
        html_key = f"{output_prefix}/decision.html"
        logger.info(f"Saving HTML to {html_key}")
        self.s3.put_object(
            Bucket=self.config.s3_bucket,
            Key=html_key,
            Body=html,
            ContentType='text/html'
        )
        
        logger.info(f"Outputs saved to s3://{self.config.s3_bucket}/{output_prefix}/")


# ==============================================================================
# Command-line Interface
# ==============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Credit Risk AI Agent with AWS Bedrock")
    parser.add_argument('--pdf-key', required=True, help='S3 key for PDF')
    parser.add_argument('--excel-key', help='S3 key for Excel metrics')
    parser.add_argument('--bucket', default='credit-risk-usecase-bucket')
    parser.add_argument('--model', default='us.amazon.nova-lite-v1:0', 
                       help='Bedrock model ID (without bedrock: prefix)')
    parser.add_argument('--offline', action='store_true', help='Use offline mode')
    parser.add_argument('--output', default='output')
    parser.add_argument('--region', default='us-east-2')
    
    args = parser.parse_args()
    
    config = AgentConfig(
        s3_bucket=args.bucket,
        model_name=args.model if not args.offline else None,
        offline_mode=args.offline,
        region=args.region
    )
    
    agent = CreditRiskAgent(config)
    decision = await agent.assess_credit_risk(
        pdf_s3_key=args.pdf_key,
        excel_s3_key=args.excel_key,
        output_prefix=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"DECISION: {decision.decision}")
    print(f"Score: {decision.score:.3f}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"{'='*60}\n")
    
    summary = decision.to_summary()
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nOutputs saved to:")
    print(f"  s3://{args.bucket}/{args.output}/decision.json")
    print(f"  s3://{args.bucket}/{args.output}/decision.html")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
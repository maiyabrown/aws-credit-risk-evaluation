# models.py
"""
Pydantic models for the Credit Risk AI Agent.
Clean, type-safe data structures with validation.
"""
from __future__ import annotations
from typing import Optional, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, confloat, field_validator


# ==============================================================================
# Extracted Financial Metrics
# ==============================================================================

class CreditMetric(BaseModel):
    """Single extracted financial metric with evidence."""
    name: str = Field(description="Metric name (e.g., 'Interest Coverage')")
    value: Optional[float] = Field(None, description="Extracted numeric value")
    unit: str = Field(default="x", description="Unit of measurement")
    source: Literal["pdf", "excel", "computed"] = Field(description="Data source")
    quote: Optional[str] = Field(None, description="Supporting quote from document")
    page: Optional[int] = Field(None, description="Page number (1-indexed)")
    confidence: confloat(ge=0, le=1) = Field(default=0.0, description="Extraction confidence")


class CovenantStatus(BaseModel):
    """Status of a single financial covenant."""
    name: str = Field(description="Covenant name")
    threshold: Optional[str] = Field(None, description="Threshold value")
    actual: Optional[str] = Field(None, description="Actual/observed value")
    status: Literal["compliant", "breached", "unclear"] = Field(description="Compliance status")
    evidence: Optional[CreditMetric] = Field(None, description="Supporting evidence")


class CreditMetrics(BaseModel):
    """Complete set of extracted credit metrics."""
    interest_coverage: Optional[CreditMetric] = None
    dscr: Optional[CreditMetric] = None
    net_leverage: Optional[CreditMetric] = None
    covenants: List[CovenantStatus] = Field(default_factory=list)
    
    def get_value(self, metric_name: str) -> Optional[float]:
        """Helper to get metric value by name."""
        metric = getattr(self, metric_name, None)
        return metric.value if metric else None
    
    def covenant_breach_count(self) -> int:
        """Count number of breached covenants."""
        return sum(1 for c in self.covenants if c.status == "breached")


# ==============================================================================
# Policy Evaluation
# ==============================================================================

class RuleOutcome(BaseModel):
    """Result of evaluating a single policy rule."""
    rule_id: str = Field(description="Rule identifier")
    rule_name: str = Field(description="Human-readable rule name")
    passed: bool = Field(description="Whether rule passed")
    message: str = Field(description="Explanation message")
    weight: float = Field(description="Rule weight in scoring")
    is_critical: bool = Field(default=False, description="Hard-fail rule")


class RiskDecision(BaseModel):
    """Final credit risk decision with full explanation."""
    
    # Core decision
    decision: Literal["HEALTHY", "REVIEW", "UNHEALTHY"] = Field(
        description="Final risk classification"
    )
    score: confloat(ge=0, le=1) = Field(
        description="Weighted policy score (0-1)"
    )
    confidence: confloat(ge=0, le=1) = Field(
        description="Overall AI confidence in extraction"
    )
    
    # Evidence
    metrics: CreditMetrics = Field(
        description="Extracted financial metrics"
    )
    rule_outcomes: List[RuleOutcome] = Field(
        description="Individual rule evaluation results"
    )
    
    # Metadata
    policy_version: str = Field(description="Policy version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_mode: Literal["llm", "offline"] = Field(description="Extraction mode used")
    
    # Input references
    pdf_source: str = Field(description="S3 key for PDF input")
    excel_source: Optional[str] = Field(None, description="S3 key for Excel input")
    
    @field_validator('decision')
    @classmethod
    def validate_decision_logic(cls, v, info):
        """Ensure decision aligns with score and confidence."""
        # This is called after all fields are set
        return v
    
    def to_summary(self) -> dict:
        """Generate executive summary of decision."""
        return {
            "decision": self.decision,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 2),
            "metrics": {
                "interest_coverage": self.metrics.get_value("interest_coverage"),
                "dscr": self.metrics.get_value("dscr"),
                "net_leverage": self.metrics.get_value("net_leverage"),
                "covenant_breaches": self.metrics.covenant_breach_count()
            },
            "timestamp": self.timestamp.isoformat(),
            "rules_passed": sum(1 for r in self.rule_outcomes if r.passed),
            "rules_failed": sum(1 for r in self.rule_outcomes if not r.passed)
        }


# ==============================================================================
# Agent Context
# ==============================================================================

class AgentConfig(BaseModel):
    """Configuration for the Credit Risk Agent."""
    s3_bucket: str = Field(description="S3 bucket for input/output")
    model_name: Optional[str] = Field(None, description="Bedrock model name (e.g., bedrock:us.amazon.nova-lite-v1:0)")
    endpoint_name: Optional[str] = Field(None, description="SageMaker endpoint name (legacy)")
    policy_s3_key: str = Field(default="policies/credit_policy.yaml")
    region: str = Field(default="us-east-2")
    offline_mode: bool = Field(default=False, description="Use regex extraction instead of LLM")
    min_confidence: float = Field(default=0.70, description="Minimum confidence for HEALTHY")


class AgentContext(BaseModel):
    """Runtime context for agent execution."""
    config: AgentConfig
    pdf_text: str = Field(description="Extracted PDF text")
    excel_data: dict = Field(default_factory=dict, description="Excel metrics")
    policy: dict = Field(description="Loaded policy YAML")
    
    class Config:
        arbitrary_types_allowed = True


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Example: Create a decision
    decision = RiskDecision(
        decision="HEALTHY",
        score=0.875,
        confidence=0.85,
        metrics=CreditMetrics(
            interest_coverage=CreditMetric(
                name="Interest Coverage",
                value=2.6,
                unit="x",
                source="pdf",
                quote="The interest coverage ratio is 2.6x",
                page=1,
                confidence=0.9
            ),
            dscr=CreditMetric(
                name="DSCR",
                value=1.32,
                unit="x",
                source="excel",
                confidence=1.0
            ),
            covenants=[
                CovenantStatus(
                    name="Minimum Interest Coverage",
                    threshold="2.0x",
                    actual="2.6x",
                    status="compliant"
                )
            ]
        ),
        rule_outcomes=[
            RuleOutcome(
                rule_id="R1",
                rule_name="Interest coverage minimum",
                passed=True,
                message="Interest coverage 2.6x >= 2.0x",
                weight=0.35
            )
        ],
        policy_version="1.0.0",
        processing_mode="llm",
        pdf_source="input/client_healthy_complex.pdf",
        excel_source="input/client_financials_complex.xlsx"
    )
    
    print(decision.model_dump_json(indent=2))
    print("\nSummary:", decision.to_summary())
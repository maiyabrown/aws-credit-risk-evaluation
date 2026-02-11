# extractors_bedrock.py
"""
Document extraction using AWS Bedrock (serverless LLM).
Supports Amazon Nova Lite and other Bedrock models via PydanticAI.
"""
from __future__ import annotations
import re
import logging
from typing import Optional
import boto3
import json

from pydantic_ai import Agent
from models import (
    AgentConfig,
    AgentContext,
    CreditMetrics,
    CreditMetric,
    CovenantStatus
)

logger = logging.getLogger(__name__)


class BedrockExtractor:
    """Extracts financial metrics using AWS Bedrock."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        if not config.offline_mode:
            # Initialize Bedrock runtime client
            self.bedrock = boto3.client(
                'bedrock-runtime',
                region_name=config.region
            )
            self.model_id = self._get_model_id(config.model_name)
            logger.info(f"Initialized Bedrock extractor with model: {self.model_id}")
        else:
            self.bedrock = None
    
    def _get_model_id(self, model_name: Optional[str]) -> str:
        """Convert PydanticAI model name to Bedrock model ID."""
        if not model_name:
            return "us.amazon.nova-lite-v1:0"
        
        # Remove "bedrock:" prefix if present
        if model_name.startswith("bedrock:"):
            return model_name.replace("bedrock:", "")
        
        return model_name
    
    async def extract_metrics(self, context: AgentContext) -> CreditMetrics:
        """Extract all credit metrics from documents."""
        
        if self.config.offline_mode:
            logger.info("Using offline regex extraction")
            return self._extract_offline(context)
        else:
            logger.info(f"Using Bedrock extraction with {self.model_id}")
            return await self._extract_with_bedrock(context)
    
    def _extract_offline(self, context: AgentContext) -> CreditMetrics:
        """Regex-based extraction (fast, no LLM needed)."""
        
        text = context.pdf_text
        
        # Extract Interest Coverage
        ic_match = re.search(
            r'interest\s+coverage.*?(\d+\.?\d*)\s*x',
            text,
            re.IGNORECASE
        )
        interest_coverage = None
        if ic_match:
            interest_coverage = CreditMetric(
                name="Interest Coverage",
                value=float(ic_match.group(1)),
                unit="x",
                source="pdf",
                quote=self._get_quote(text, ic_match.start(), ic_match.end()),
                confidence=0.95
            )
        
        # Get from Excel
        dscr = None
        net_leverage = None
        if context.excel_data:
            if 'DSCR' in context.excel_data:
                dscr = CreditMetric(
                    name="DSCR",
                    value=float(context.excel_data['DSCR']),
                    unit="x",
                    source="excel",
                    confidence=1.0
                )
            if 'NetLeverage' in context.excel_data:
                net_leverage = CreditMetric(
                    name="Net Leverage",
                    value=float(context.excel_data['NetLeverage']),
                    unit="x",
                    source="excel",
                    confidence=1.0
                )
        
        # Extract covenants
        covenants = self._extract_covenants_offline(text)
        
        return CreditMetrics(
            interest_coverage=interest_coverage,
            dscr=dscr,
            net_leverage=net_leverage,
            covenants=covenants
        )
    
    def _extract_covenants_offline(self, text: str) -> list[CovenantStatus]:
        """Extract covenant compliance using regex patterns."""
        covenants = []
        
        # Pattern for structured covenant statements
        pattern = r'(Minimum|Maximum)\s+([\w\s]+):\s*threshold\s+([\d.]+)x;\s*reported\s+([\d.]+)x\s*—\s*(\w+)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            min_max = match.group(1)
            name = match.group(2).strip()
            threshold = match.group(3)
            actual = match.group(4)
            status_text = match.group(5).lower()
            
            status = "compliant" if "compliant" in status_text else "breached"
            
            covenants.append(CovenantStatus(
                name=f"{min_max} {name}",
                threshold=f"{threshold}x",
                actual=f"{actual}x",
                status=status,
                evidence=CreditMetric(
                    name=name,
                    value=float(actual),
                    unit="x",
                    source="pdf",
                    confidence=0.9
                )
            ))
        
        # Fallback: general compliance statements
        if not covenants:
            if re.search(r'all.*covenant.*compliant', text, re.IGNORECASE):
                covenants.append(CovenantStatus(
                    name="All Covenants",
                    status="compliant"
                ))
            elif re.search(r'breach|violation|not in compliance', text, re.IGNORECASE):
                covenants.append(CovenantStatus(
                    name="Covenant Breach Detected",
                    status="breached"
                ))
        
        return covenants
    
    async def _extract_with_bedrock(self, context: AgentContext) -> CreditMetrics:
        """Extract using AWS Bedrock (Amazon Nova)."""
        
        text = context.pdf_text[:5000]  # Limit for context
        
        # Extract Interest Coverage using Bedrock
        ic_prompt = f"""Extract the interest coverage ratio from this financial document.

Document: {text}

Respond with ONLY the number followed by 'x'. For example: 2.6x

Interest coverage:"""
        
        ic_response = await self._call_bedrock(ic_prompt)
        
        interest_coverage = None
        if ic_response:
            match = re.search(r'(\d+\.?\d*)', ic_response)
            if match:
                interest_coverage = CreditMetric(
                    name="Interest Coverage",
                    value=float(match.group(1)),
                    unit="x",
                    source="pdf",
                    confidence=0.85
                )
        
        # Get DSCR and Net Leverage from Excel (more reliable)
        dscr = None
        net_leverage = None
        if context.excel_data:
            if 'DSCR' in context.excel_data:
                dscr = CreditMetric(
                    name="DSCR",
                    value=float(context.excel_data['DSCR']),
                    unit="x",
                    source="excel",
                    confidence=1.0
                )
            if 'NetLeverage' in context.excel_data:
                net_leverage = CreditMetric(
                    name="Net Leverage",
                    value=float(context.excel_data['NetLeverage']),
                    unit="x",
                    source="excel",
                    confidence=1.0
                )
        
        # Extract covenant status
        cov_prompt = f"""Are there any financial covenant breaches in this document?

Document: {text}

Respond with ONLY 'yes' or 'no'.

Answer:"""
        
        cov_response = await self._call_bedrock(cov_prompt)
        
        covenants = []
        if cov_response and 'yes' in cov_response.lower():
            covenants.append(CovenantStatus(
                name="Covenant Breach",
                status="breached"
            ))
        elif cov_response and 'no' in cov_response.lower():
            covenants.append(CovenantStatus(
                name="All Covenants",
                status="compliant"
            ))
        
        return CreditMetrics(
            interest_coverage=interest_coverage,
            dscr=dscr,
            net_leverage=net_leverage,
            covenants=covenants
        )
    
    async def _call_bedrock(self, prompt: str) -> Optional[str]:
        """Call Bedrock model with a prompt."""
        if not self.bedrock:
            return None
        
        try:
            # Format request for Amazon Nova
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 100,
                    "temperature": 0.0,
                    "topP": 0.9
                }
            }
            
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            result = json.loads(response['body'].read())
            
            # Extract text from response
            if 'output' in result and 'message' in result['output']:
                content = result['output']['message'].get('content', [])
                if content and len(content) > 0:
                    return content[0].get('text', '')
            
            return None
            
        except Exception as e:
            logger.error(f"Bedrock extraction failed: {e}")
            return None
    
    @staticmethod
    def _get_quote(text: str, start: int, end: int, padding: int = 60) -> str:
        """Extract a quote with context."""
        quote_start = max(0, start - padding)
        quote_end = min(len(text), end + padding)
        return text[quote_start:quote_end].strip()


# Alias for compatibility
DocumentExtractor = BedrockExtractor
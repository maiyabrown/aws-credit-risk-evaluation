# policy.py
"""
Policy evaluation engine for credit risk assessment.
Evaluates extracted metrics against human-governed policy rules.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List

from models import CreditMetrics, RuleOutcome

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Evaluates credit metrics against policy rules."""
    
    def evaluate(
        self,
        metrics: CreditMetrics,
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate metrics against policy rules.
        
        Returns:
            dict with 'score', 'outcomes', 'hard_fail_triggered', 'healthy_threshold'
        """
        thresholds = policy.get('thresholds', {})
        rules = policy.get('rules', [])
        aggregation = policy.get('aggregation', {})
        
        outcomes: List[RuleOutcome] = []
        score = 0.0
        weight_sum = 0.0
        hard_fail_triggered = False
        
        # Build evaluation context
        context = self._build_context(metrics, thresholds)
        
        # Evaluate each rule
        for rule in rules:
            outcome = self._evaluate_rule(rule, context)
            outcomes.append(outcome)
            
            if outcome.passed:
                score += outcome.weight
            weight_sum += outcome.weight
            
            if outcome.is_critical and not outcome.passed:
                hard_fail_triggered = True
        
        # Normalize score
        normalized_score = (score / weight_sum) if weight_sum > 0 else 1.0
        
        return {
            'score': normalized_score,
            'outcomes': outcomes,
            'hard_fail_triggered': hard_fail_triggered,
            'healthy_threshold': float(aggregation.get('healthy_threshold', 0.70))
        }
    
    def _build_context(
        self,
        metrics: CreditMetrics,
        thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build evaluation context with metrics and thresholds."""
        return {
            'interest_coverage': metrics.get_value('interest_coverage'),
            'dscr': metrics.get_value('dscr'),
            'net_leverage': metrics.get_value('net_leverage'),
            'covenant_breach_count': metrics.covenant_breach_count(),
            'thresholds': thresholds
        }
    
    def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RuleOutcome:
        """Evaluate a single policy rule."""
        
        rule_id = rule['id']
        rule_name = rule.get('name', rule_id)
        weight = float(rule.get('weight', 0.0))
        is_critical = bool(rule.get('hard_fail', False))
        
        # Check if rule should be evaluated
        when_condition = rule.get('when')
        if when_condition:
            should_evaluate = self._safe_eval(when_condition, context)
            if not should_evaluate:
                # Rule doesn't apply, treat as passed
                return RuleOutcome(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    passed=True,
                    message=f"{rule_name}: Not applicable",
                    weight=weight,
                    is_critical=is_critical
                )
        
        # Evaluate main condition
        condition = rule.get('condition', 'False')
        passed = self._safe_eval(condition, context)
        
        # Generate message
        message_template = rule.get('pass' if passed else 'fail', '')
        message = self._render_message(message_template, context)
        
        return RuleOutcome(
            rule_id=rule_id,
            rule_name=rule_name,
            passed=passed,
            message=message,
            weight=weight,
            is_critical=is_critical
        )
    
    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a boolean expression."""
        try:
            # Create safe evaluation environment
            safe_context = {
                'features': context,
                'thresholds': context.get('thresholds', {})
            }
            
            # Replace dot notation with dictionary access
            # e.g., "features.dscr >= thresholds.dscr_min" 
            # -> "features['dscr'] >= thresholds['dscr_min']"
            modified_expr = expression
            for key in ['features', 'thresholds']:
                import re
                pattern = f"{key}\.(\w+)"
                modified_expr = re.sub(
                    pattern,
                    lambda m: f"{key}['{m.group(1)}']",
                    modified_expr
                )
            
            result = eval(modified_expr, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Rule evaluation error: {expression} - {e}")
            return False
    
    def _render_message(self, template: str, context: Dict[str, Any]) -> str:
        """Render message template with context values."""
        try:
            from jinja2 import Template
            
            # Build template context
            tpl_context = {
                'features': context,
                'thresholds': context.get('thresholds', {})
            }
            
            return Template(template).render(**tpl_context)
            
        except Exception as e:
            logger.error(f"Message rendering error: {e}")
            return template


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    from models import CreditMetric
    
    # Sample policy
    policy = {
        'version': '1.0.0',
        'thresholds': {
            'interest_coverage_min': 2.0,
            'dscr_min': 1.25,
            'net_leverage_max': 3.5
        },
        'rules': [
            {
                'id': 'R1',
                'name': 'Interest Coverage Minimum',
                'when': "features['interest_coverage'] is not None",
                'condition': "features['interest_coverage'] >= thresholds['interest_coverage_min']",
                'pass': "Interest coverage {{features['interest_coverage']}}x >= {{thresholds['interest_coverage_min']}}x",
                'fail': "Interest coverage {{features['interest_coverage']}}x < {{thresholds['interest_coverage_min']}}x",
                'weight': 0.35
            },
            {
                'id': 'R2',
                'name': 'DSCR Minimum',
                'when': "features['dscr'] is not None",
                'condition': "features['dscr'] >= thresholds['dscr_min']",
                'pass': "DSCR {{features['dscr']}}x >= {{thresholds['dscr_min']}}x",
                'fail': "DSCR {{features['dscr']}}x < {{thresholds['dscr_min']}}x",
                'weight': 0.35
            },
            {
                'id': 'R4',
                'name': 'No Covenant Breaches',
                'condition': "features['covenant_breach_count'] == 0",
                'pass': "No covenant breaches",
                'fail': "{{features['covenant_breach_count']}} covenant breach(es)",
                'weight': 0.10,
                'hard_fail': True
            }
        ],
        'aggregation': {
            'healthy_threshold': 0.70
        }
    }
    
    # Sample metrics
    metrics = CreditMetrics(
        interest_coverage=CreditMetric(name="IC", value=2.6, unit="x", source="pdf", confidence=0.9),
        dscr=CreditMetric(name="DSCR", value=1.32, unit="x", source="excel", confidence=1.0),
        net_leverage=CreditMetric(name="NL", value=3.2, unit="x", source="excel", confidence=1.0)
    )
    
    # Evaluate
    engine = PolicyEngine()
    result = engine.evaluate(metrics, policy)
    
    print(f"Score: {result['score']:.3f}")
    print(f"Hard Fail: {result['hard_fail_triggered']}")
    print("\nRule Outcomes:")
    for outcome in result['outcomes']:
        status = "✓" if outcome.passed else "✗"
        print(f"{status} {outcome.rule_name}: {outcome.message}")
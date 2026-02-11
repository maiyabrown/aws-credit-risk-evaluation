# AWS Bedrock Setup Guide

## ✨ Why Bedrock Over SageMaker Endpoints?

### Bedrock Advantages
- ✅ **Serverless** - No endpoints to manage or deploy
- ✅ **Always Ready** - No cold start, no timeouts
- ✅ **Pay-per-use** - Only pay for what you use
- ✅ **No GPU quotas needed** - Works immediately
- ✅ **Native PydanticAI support** - Works out of the box

### Cost Comparison

| Approach | Setup | Cost | Speed | Reliability |
|----------|-------|------|-------|-------------|
| **SageMaker Endpoint** | Deploy & wait 10min | $1.41/hr (24/7) = $1,010/month | Fast | Can timeout |
| **AWS Bedrock Nova Lite** | Enable & use | $0.075 per 1M tokens = $0.0001 per doc | Very Fast | Always ready |

**Break-even**: ~600,000 documents/month

**For typical usage (100 docs/day)**: Bedrock saves **$1,000/month**!

## 🚀 Setup (5 minutes)

### Step 1: Enable Bedrock Model Access (deprecated)

```bash
# Open AWS Console
# Navigate to: Bedrock → Model Access → Manage Model Access
# Enable: Amazon Nova Lite (us.amazon.nova-lite-v1:0)
```

Or via CLI:
```bash
aws bedrock get-foundation-model \
    --model-identifier us.amazon.nova-lite-v1:0 \
    --region us-east-2
```

### Step 2: Update IAM Role

Your execution role needs Bedrock permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-2::foundation-model/us.amazon.nova-lite-v1:0"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::credit-risk-usecase-bucket",
                "arn:aws:s3:::credit-risk-usecase-bucket/*"
            ]
        }
    ]
}
```

### Step 3: Install Updated Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Assessment

```bash
python agent_bedrock.py \
    --pdf-key input/documents/client_healthy_complex.pdf \
    --excel-key input/metrics/client_financials_complex.xlsx \
    --bucket credit-risk-usecase-bucket \
    --model bedrock:us.amazon.nova-lite-v1:0 \
    --region us-east-2 \
    --output output/bedrock-test
```

That's it! No endpoint deployment needed!

## 📊 Available Bedrock Models

### Recommended for Credit Risk

| Model | Model ID | Cost (per 1M tokens) | Best For |
|-------|----------|---------------------|----------|
| **Amazon Nova Lite** ⭐ | `us.amazon.nova-lite-v1:0` | $0.075 | Fast extraction, cost-effective |
| **Amazon Nova Pro** | `us.amazon.nova-pro-v1:0` | $0.80 | Higher accuracy |
| **Claude 3.5 Haiku** | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | $1.00 | Best reasoning |
| **Llama 3.3 70B** | `us.meta.llama3-3-70b-instruct-v1:0` | $0.99 | Open source |

### Usage Examples

**Amazon Nova Lite (Recommended):**
```python
config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    model_name="bedrock:us.amazon.nova-lite-v1:0",
    region="us-east-2"
)
```

**Claude 3.5 Haiku (Best accuracy):**
```python
config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    model_name="bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region="us-east-2"
)
```

**Offline Mode (Free, no LLM):**
```python
config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    offline_mode=True
)
```

## 🔧 Configuration

### Environment Variables

```bash
export AWS_DEFAULT_REGION=us-east-2
export CREDIT_RISK_BUCKET=credit-risk-usecase-bucket
export CREDIT_RISK_MODEL=bedrock:us.amazon.nova-lite-v1:0
```

### Python Code

```python
from agent import CreditRiskAgent
from models import AgentConfig

config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    model_name="bedrock:us.amazon.nova-lite-v1:0",
    region="us-east-2",
    offline_mode=False,
    min_confidence=0.70
)

agent = CreditRiskAgent(config)

# Run assessment
result = await agent.assess_credit_risk(
    pdf_s3_key="input/documents/client_healthy_complex.pdf",
    excel_s3_key="input/metrics/client_financials_complex.xlsx"
)

print(f"Decision: {result.decision}")
print(f"Score: {result.score}")
```

## 💰 Cost Estimation

### Per Document Costs

Assuming:
- PDF: 2,000 tokens input
- Response: 200 tokens output
- Nova Lite: $0.075 / 1M input, $0.30 / 1M output

**Cost per document:**
```
Input:  2,000 × $0.075 / 1,000,000 = $0.00015
Output:   200 × $0.30  / 1,000,000 = $0.00006
Total:                               $0.00021 (~$0.0002)
```

### Monthly Costs

| Documents/Month | Bedrock Cost | SageMaker Endpoint | Savings |
|-----------------|--------------|-------------------|---------|
| 100 | $0.02 | $1,010 | $1,010 |
| 1,000 | $0.21 | $1,010 | $1,010 |
| 10,000 | $2.10 | $1,010 | $1,008 |
| 100,000 | $21.00 | $1,010 | $989 |
| 1,000,000 | $210.00 | $1,010 | $800 |

**Bedrock is cheaper until you hit 5+ million documents/month!**

## 🎯 Testing

### Test Bedrock Access
```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-2')

body = {
    "messages": [{"role": "user", "content": [{"text": "What is 2+2?"}]}],
    "inferenceConfig": {"maxTokens": 10, "temperature": 0}
}

response = bedrock.invoke_model(
    modelId='us.amazon.nova-lite-v1:0',
    body=json.dumps(body)
)

result = json.loads(response['body'].read())
print(result)
```

### Test Full Pipeline
```bash
# Offline mode (no Bedrock needed)
python agent_bedrock.py \
    --pdf-key input/documents/client_healthy_complex.pdf \
    --bucket credit-risk-usecase-bucket \
    --offline

# Bedrock mode
python agent_bedrock.py \
    --pdf-key input/documents/client_healthy_complex.pdf \
    --bucket credit-risk-usecase-bucket \
    --model bedrock:us.amazon.nova-lite-v1:0
```

## 🔍 Monitoring

### CloudWatch Metrics

Bedrock automatically logs:
- Number of invocations
- Input/output tokens
- Latency
- Errors

View in CloudWatch:
```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/Bedrock \
    --metric-name InvocationCount \
    --start-time 2026-02-01T00:00:00Z \
    --end-time 2026-02-10T23:59:59Z \
    --period 3600 \
    --statistics Sum \
    --region us-east-2
```

### Cost Tracking

```bash
# View Bedrock costs
aws ce get-cost-and-usage \
    --time-period Start=2026-02-01,End=2026-02-10 \
    --granularity DAILY \
    --metrics UnblendedCost \
    --filter file://bedrock-filter.json
```

bedrock-filter.json:
```json
{
    "Dimensions": {
        "Key": "SERVICE",
        "Values": ["Amazon Bedrock"]
    }
}
```

## 🚀 Deployment Options

### Lambda Function

```python
import json
from agent import CreditRiskAgent
from models import AgentConfig

def lambda_handler(event, context):
    config = AgentConfig(
        s3_bucket=event['bucket'],
        model_name="bedrock:us.amazon.nova-lite-v1:0",
        region="us-east-2"
    )
    
    agent = CreditRiskAgent(config)
    
    result = await agent.assess_credit_risk(
        pdf_s3_key=event['pdf_key'],
        excel_s3_key=event.get('excel_key')
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(result.to_summary())
    }
```

### SageMaker Processing Job

```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    role=role,
    image_uri="pytorch-cpu-image",  # CPU is fine with Bedrock
    instance_type="ml.t3.medium",  # Cheap instance
    instance_count=1
)

processor.run(
    code="agent_bedrock.py",
    arguments=[
        "--pdf-key", "input/documents/client_report.pdf",
        "--bucket", "credit-risk-usecase-bucket",
        "--model", "bedrock:us.amazon.nova-lite-v1:0"
    ]
)
```

## 📋 Migration from SageMaker Endpoint

### Before (SageMaker)
```python
config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    endpoint_name="finance-llm-endpoint",  # ❌ Requires deployment
    region="us-east-2"
)
```

### After (Bedrock)
```python
config = AgentConfig(
    s3_bucket="credit-risk-usecase-bucket",
    model_name="bedrock:us.amazon.nova-lite-v1:0",  # ✅ Serverless!
    region="us-east-2"
)
```

**That's it!** Everything else stays the same.

## 🎓 Best Practices

1. **Start with Nova Lite** - Fast and cheap
2. **Use offline mode for development** - Free!
3. **Monitor token usage** - Optimize prompts
4. **Cache results** - For duplicate documents
5. **Batch when possible** - Process multiple docs in one job

## 🆘 Troubleshooting

### "Model not found"
→ Enable model access in Bedrock console

### "Access Denied"
→ Add Bedrock permissions to IAM role

### "Throttling"
→ Bedrock has generous limits, but you can request increases

### Still want SageMaker?
→ Use `agent.py` and `extractors.py` instead of the `_bedrock` versions

## 🎉 Summary

**Bedrock Setup: 5 minutes**
- Enable model access
- Add IAM permissions
- Run code

**No deployment, no waiting, no endpoints, no timeouts!**
# Credit Risk AI Agent

This project use an AI Agent to determine the credit risk of a client. This guide explains how to deploy the credit risk assessment pipeline on AWS SageMaker.

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

## Prerequisites

1. AWS Account with appropriate permissions
2. S3 bucket: `credit-risk-usecase-bucket`
3. SageMaker execution role with S3 access
4. AWS CLI configured locally

## Project Structure

```
credit-risk-agent/
modules
│   ├── agent.py                 # Main Bedrock Agent
│   ├── models.py                # Pydantic models for data 
│   ├── extractors.py            # Document extraction logic
│   ├── policy.py                # Policy evaluation engine
├── policies/
│   └── credit_policy.yaml       # Human-governed credit policy
├── requirements.txt
└── README.md
```

## S3 Bucket Structure

Organize your S3 bucket as follows:

```
s3://credit-risk-usecase-bucket/
├── policies/
│   └── credit_policy.yaml          # Policy configuration
├── input/
│   ├── documents/                  # PDF documents to process
│   │   └── company_financials.pdf
│   └── metrics/                    # Excel files with metrics
│       └── company_metrics.xlsx
└── output/                         # Processing results
    └── <timestamp>/
        ├── decision.json
        └── decision.html
```

## Data Flow

```
S3 Input (PDF + Excel)
    ↓
Extract Text/Tables
    ↓
Bedrock Agent
    ├── Access Amazon Nova Lite
    ├── Load Policy (YAML)
    └── Evaluate Rules
    ↓
Decision Object (with evidence)
    ↓
S3 Output (JSON + HTML)
```

### Install Updated Dependencies

```bash
pip install -r requirements.txt
```

### Run Assessment

**Online Mode (Bedrock Agent):**
```bash
python agent_bedrock.py \
    --pdf-key input/documents/client_healthy_complex.pdf \
    --excel-key input/metrics/client_financials_complex.xlsx \
    --bucket credit-risk-usecase-bucket \
    --model bedrock:us.amazon.nova-lite-v1:0 \
    --region us-east-2 \
    --output output/bedrock-test
```
**Offline Mode (Free, no LLM):**
```bash
python agent_bedrock.py \
    --pdf-key input/documents/client_healthy_complex.pdf \
    --excel-key input/metrics/client_financials_complex.xlsx \
    --bucket credit-risk-usecase-bucket \
    --region us-east-2 \
    --output output/offline-test
    --offline
```

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

### Monthly Costs

| Documents/Month | Bedrock Cost | SageMaker Endpoint | Savings |
|-----------------|--------------|-------------------|---------|
| 100 | $0.02 | $1,010 | $1,010 |
| 1,000 | $0.21 | $1,010 | $1,010 |
| 10,000 | $2.10 | $1,010 | $1,008 |
| 100,000 | $21.00 | $1,010 | $989 |
| 1,000,000 | $210.00 | $1,010 | $800 |

**Bedrock is cheaper until you hit 5+ million documents/month!**

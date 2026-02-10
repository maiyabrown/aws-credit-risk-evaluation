# Example Usage and Configuration

## Quick Start Examples

### 1. Basic Usage (Offline Mode - No Model Downloads)
```bash
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/company_report.pdf \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/test_run \
    --offline
```

### 2. With Excel Metrics
```bash
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/company_report.pdf \
    --excel-key input/metrics/company_metrics.xlsx \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/test_run \
    --offline
```

### 3. Using LLM (Online Mode)
```bash
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/company_report.pdf \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/test_run \
    --text-llm AdaptLLM/finance-LLM \
    --region us-east-1
```

## SageMaker Processing Job Examples

### 1. Launch Single Processing Job
```bash
python sagemaker_processing_job.py \
    --pdf-key input/documents/company_report.pdf \
    --bucket credit-risk-usecase-bucket \
    --instance-type ml.m5.xlarge
```

### 2. With Custom Instance Type
```bash
python sagemaker_processing_job.py \
    --pdf-key input/documents/company_report.pdf \
    --excel-key input/metrics/company_metrics.xlsx \
    --bucket credit-risk-usecase-bucket \
    --instance-type ml.m5.2xlarge
```

### 3. GPU Instance for Faster LLM Processing
```bash
python sagemaker_processing_job.py \
    --pdf-key input/documents/company_report.pdf \
    --bucket credit-risk-usecase-bucket \
    --instance-type ml.g4dn.xlarge \
    --online
```

## Python API Usage

### Direct Function Call
```python
from risk_pipeline_sagemaker import S3Handler, process_risk_assessment

# Initialize S3 handler
s3_handler = S3Handler("credit-risk-usecase-bucket", region_name="us-east-1")

# Process a single document
result = process_risk_assessment(
    s3_handler=s3_handler,
    pdf_s3_key="input/documents/company_report.pdf",
    excel_s3_key="input/metrics/company_metrics.xlsx",
    policy_s3_key="config/credit_policy.yaml",
    output_prefix="output/my_analysis",
    text_llm="AdaptLLM/finance-LLM",
    offline=True  # Set False to use LLM
)

print(f"Decision: {result['decision_label']}")
print(f"Score: {result['decision_score']}")
print(f"Results: {result['json_s3_key']}")
```

### Batch Processing
```python
from risk_pipeline_sagemaker import S3Handler, process_risk_assessment
from concurrent.futures import ThreadPoolExecutor

s3_handler = S3Handler("credit-risk-usecase-bucket")

# Get list of PDFs to process
pdf_files = s3_handler.list_files("input/documents/")
pdf_files = [f for f in pdf_files if f.endswith('.pdf')]

def process_one(pdf_key):
    doc_id = pdf_key.split('/')[-1].replace('.pdf', '')
    return process_risk_assessment(
        s3_handler=s3_handler,
        pdf_s3_key=pdf_key,
        excel_s3_key=None,
        policy_s3_key="config/credit_policy.yaml",
        output_prefix=f"output/{doc_id}",
        text_llm="AdaptLLM/finance-LLM",
        offline=True
    )

# Process in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_one, pdf_files))

print(f"Processed {len(results)} documents")
```

## Environment Variables

You can set default values using environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export CREDIT_RISK_BUCKET=credit-risk-usecase-bucket
export CREDIT_RISK_POLICY_KEY=config/credit_policy.yaml

# Then run with fewer arguments
python risk_pipeline_sagemaker.py \
    --pdf-key input/documents/company_report.pdf \
    --offline
```

## Testing Locally Before SageMaker

Test the pipeline locally first:

```bash
# 1. Set up local test files
mkdir -p test_data
aws s3 cp s3://credit-risk-usecase-bucket/input/documents/sample.pdf test_data/
aws s3 cp s3://credit-risk-usecase-bucket/config/credit_policy.yaml test_data/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test locally (modify paths in code temporarily or use local file handling)
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/sample.pdf \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/local_test \
    --offline
```

## Monitoring and Debugging

### View Processing Job Logs
```bash
# List recent processing jobs
aws sagemaker list-processing-jobs \
    --name-contains credit-risk \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 5

# View CloudWatch logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow
```

### Check Output Files
```bash
# List output files
aws s3 ls s3://credit-risk-usecase-bucket/output/ --recursive

# Download results
aws s3 cp s3://credit-risk-usecase-bucket/output/20240210_120000/decision.json .
aws s3 cp s3://credit-risk-usecase-bucket/output/20240210_120000/decision.html .

# View in browser (macOS)
open decision.html

# View JSON
cat decision.json | jq .
```

## Expected Output Structure

### decision.json
```json
{
  "policy_version": "1.0.0",
  "pdf_s3_key": "input/documents/company_report.pdf",
  "excel_s3_key": "input/metrics/company_metrics.xlsx",
  "features": {
    "interest_coverage": 2.3,
    "dscr": 1.28,
    "net_leverage": 3.2,
    "covenant_breach_count": 0
  },
  "decision": {
    "label": "HEALTHY",
    "score": 0.875,
    "outcomes": [...],
    "healthy_threshold": 0.7,
    "hard_fail_triggered": false
  },
  "ai": {
    "interest_coverage": {...},
    "covenants": {...},
    "discrepancies": {...}
  }
}
```

## Common Issues and Solutions

### Issue: boto3 not found
```bash
pip install boto3 botocore
```

### Issue: Permission denied on S3
```bash
# Check IAM role permissions
aws iam get-role --role-name YourSageMakerRole

# Attach S3 policy
aws iam attach-role-policy \
    --role-name YourSageMakerRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Issue: PDF extraction fails
- Ensure PDF is not encrypted
- Check PDF is not scanned image (needs OCR)
- Verify file uploaded correctly to S3

### Issue: Out of memory
- Increase SageMaker instance size
- Reduce PDF text limit in code (currently 180000 chars)
- Process smaller documents

## Performance Tips

1. **Use offline mode** for faster processing when high accuracy isn't critical
2. **GPU instances** (ml.g4dn.xlarge) for LLM-based extraction
3. **Larger instances** (ml.m5.2xlarge) for parallel PDF processing
4. **Batch processing** to process multiple documents efficiently
5. **Pre-download models** to S3 to avoid repeated downloads

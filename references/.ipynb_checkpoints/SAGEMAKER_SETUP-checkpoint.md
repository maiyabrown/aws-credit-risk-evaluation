# AWS SageMaker Deployment Guide

This guide explains how to deploy the credit risk assessment pipeline on AWS SageMaker.

## Prerequisites

1. AWS Account with appropriate permissions
2. S3 bucket: `credit-risk-usecase-bucket`
3. SageMaker execution role with S3 access
4. AWS CLI configured locally

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
├── models/                         # Optional: Pre-downloaded HF models
│   └── finance-LLM/
└── output/                         # Processing results
    └── <timestamp>/
        ├── decision.json
        └── decision.html
```

## Setup Steps

### 1. Upload Configuration Files to S3

```bash
# Upload policy file
aws s3 cp credit_policy.yaml s3://credit-risk-usecase-bucket/config/

# Upload your documents
aws s3 cp company_financials.pdf s3://credit-risk-usecase-bucket/input/documents/

# Upload metrics (optional)
aws s3 cp company_metrics.xlsx s3://credit-risk-usecase-bucket/input/metrics/
```

### 2. Create SageMaker Notebook Instance or Use Studio

#### Option A: SageMaker Notebook Instance
```bash
aws sagemaker create-notebook-instance \
    --notebook-instance-name credit-risk-processor \
    --instance-type ml.t3.xlarge \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
```

#### Option B: SageMaker Studio
Use the AWS Console to create a SageMaker Studio domain and user profile.

### 3. Install Dependencies

In your SageMaker environment, run:

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

#### Basic usage (offline mode - no model downloads):
```bash
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/company_financials.pdf \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/$(date +%Y%m%d_%H%M%S) \
    --offline
```

#### Advanced usage (with LLM):
```bash
python risk_pipeline_sagemaker.py \
    --bucket credit-risk-usecase-bucket \
    --pdf-key input/documents/company_financials.pdf \
    --excel-key input/metrics/company_metrics.xlsx \
    --policy-key config/credit_policy.yaml \
    --output-prefix output/$(date +%Y%m%d_%H%M%S) \
    --text-llm AdaptLLM/finance-LLM \
    --region us-east-1
```

## Running as a SageMaker Processing Job

For production workloads, use SageMaker Processing:

### Create processing script:

```python
# sagemaker_processing_job.py
import sagemaker
from sagemaker.processing import ScriptProcessor

# Initialize session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name

# Create processor
processor = ScriptProcessor(
    role=role,
    image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.0-gpu-py310",
    instance_type="ml.m5.xlarge",  # Use ml.g4dn.xlarge for GPU
    instance_count=1,
    base_job_name="credit-risk-assessment",
)

# Run processing job
processor.run(
    code="risk_pipeline_sagemaker.py",
    arguments=[
        "--bucket", "credit-risk-usecase-bucket",
        "--pdf-key", "input/documents/company_financials.pdf",
        "--policy-key", "config/credit_policy.yaml",
        "--output-prefix", "output/processing-job-{date}",
        "--offline"
    ],
)
```

## Batch Processing Multiple Documents

To process multiple documents in parallel:

```python
# batch_processor.py
import boto3
from concurrent.futures import ThreadPoolExecutor
from risk_pipeline_sagemaker import S3Handler, process_risk_assessment

def process_document(s3_key: str, bucket: str = "credit-risk-usecase-bucket"):
    """Process a single document."""
    s3_handler = S3Handler(bucket)
    
    # Extract document ID from path
    doc_id = s3_key.split('/')[-1].replace('.pdf', '')
    
    result = process_risk_assessment(
        s3_handler=s3_handler,
        pdf_s3_key=s3_key,
        excel_s3_key=None,
        policy_s3_key="config/credit_policy.yaml",
        output_prefix=f"output/{doc_id}",
        text_llm="AdaptLLM/finance-LLM",
        offline=True
    )
    
    return result

# List all PDFs in input directory
s3_handler = S3Handler("credit-risk-usecase-bucket")
pdf_keys = [k for k in s3_handler.list_files("input/documents/") if k.endswith('.pdf')]

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_document, pdf_keys))

print(f"Processed {len(results)} documents")
```

## Lambda Integration (Optional)

For event-driven processing when new documents are uploaded:

```python
# lambda_function.py
import json
from risk_pipeline_sagemaker import S3Handler, process_risk_assessment

def lambda_handler(event, context):
    """Triggered when a new PDF is uploaded to S3."""
    
    # Extract S3 event information
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    pdf_key = record['s3']['object']['key']
    
    # Only process PDFs in the input directory
    if not pdf_key.startswith('input/documents/') or not pdf_key.endswith('.pdf'):
        return {'statusCode': 200, 'body': 'Skipped non-PDF file'}
    
    # Process the document
    s3_handler = S3Handler(bucket)
    doc_id = pdf_key.split('/')[-1].replace('.pdf', '')
    
    try:
        result = process_risk_assessment(
            s3_handler=s3_handler,
            pdf_s3_key=pdf_key,
            excel_s3_key=None,
            policy_s3_key="config/credit_policy.yaml",
            output_prefix=f"output/{doc_id}",
            text_llm="AdaptLLM/finance-LLM",
            offline=True
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete',
                'decision': result['decision_label'],
                'output': result['json_s3_key']
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

## IAM Policy for SageMaker Role

Your SageMaker execution role needs these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
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
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob"
            ],
            "Resource": "*"
        }
    ]
}
```

## Monitoring and Logging

View logs in CloudWatch:
```bash
aws logs tail /aws/sagemaker/ProcessingJobs --follow
```

## Cost Optimization Tips

1. **Use Offline Mode**: Set `--offline` flag to avoid downloading large HuggingFace models
2. **Instance Selection**: 
   - Use `ml.t3.xlarge` for offline/regex mode
   - Use `ml.g4dn.xlarge` for GPU-accelerated LLM inference
3. **Spot Instances**: Use SageMaker managed spot training for 70% cost savings
4. **S3 Lifecycle**: Set lifecycle policies to archive old outputs to S3 Glacier

## Troubleshooting

### Issue: Out of Memory
**Solution**: Increase instance size or reduce PDF text limit in code

### Issue: Model Download Timeout
**Solution**: Pre-download models to S3 and load from there:
```python
# In ai_assist_risk.py, modify HFJsonClient initialization
model_path = "s3://credit-risk-usecase-bucket/models/finance-LLM"
```

### Issue: S3 Permission Denied
**Solution**: Verify IAM role has correct S3 permissions

## Next Steps

1. Set up CloudWatch alarms for failed processing jobs
2. Create a Step Functions workflow for complex pipelines
3. Integrate with SNS for email notifications on high-risk decisions
4. Build a dashboard with QuickSight to visualize risk assessments over time

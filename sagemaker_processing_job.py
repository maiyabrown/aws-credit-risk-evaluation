# sagemaker_processing_job.py
"""
Script to launch SageMaker Processing Jobs for credit risk assessment.
This allows you to process documents at scale without managing infrastructure.
"""
import argparse
from datetime import datetime
import sagemaker
from sagemaker.processing import ScriptProcessor

def create_processing_job(
    pdf_s3_key: str,
    excel_s3_key: str = None,
    bucket: str = "credit-risk-usecase-bucket",
    policy_key: str = "config/credit_policy.yaml",
    instance_type: str = "ml.m5.xlarge",
    offline: bool = True,
    region: str = None
):
    """
    Create and run a SageMaker Processing Job for risk assessment.
    
    Args:
        pdf_s3_key: S3 key for the PDF document
        excel_s3_key: Optional S3 key for Excel metrics
        bucket: S3 bucket name
        policy_key: S3 key for policy YAML
        instance_type: SageMaker instance type
        offline: Use offline regex mode (no model downloads)
        region: AWS region
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    if region is None:
        region = sagemaker_session.boto_region_name
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"output/{timestamp}"
    
    # Choose appropriate container image
    if 'g4dn' in instance_type or 'p3' in instance_type:
        image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.0-gpu-py310"
    else:
        image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
    
    # Create processor
    processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        base_job_name="credit-risk-assessment",
        sagemaker_session=sagemaker_session,
        volume_size_in_gb=30,
    )
    
    # Build arguments
    arguments = [
        "--bucket", bucket,
        "--pdf-key", pdf_s3_key,
        "--policy-key", policy_key,
        "--output-prefix", output_prefix,
        "--region", region,
    ]
    
    if excel_s3_key:
        arguments.extend(["--excel-key", excel_s3_key])
    
    if offline:
        arguments.append("--offline")
    
    # Run processing job
    print(f"Launching SageMaker Processing Job...")
    print(f"  PDF: s3://{bucket}/{pdf_s3_key}")
    print(f"  Output: s3://{bucket}/{output_prefix}/")
    
    processor.run(
        code="risk_pipeline_sagemaker.py",
        arguments=arguments,
        wait=True,
        logs=True,
    )
    
    print(f"\n✅ Processing complete!")
    print(f"Results: s3://{bucket}/{output_prefix}/")
    
    return {
        'job_name': processor.latest_job.name,
        'output_s3_uri': f"s3://{bucket}/{output_prefix}/",
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Launch SageMaker Processing Jobs"
    )
    
    parser.add_argument('--pdf-key', required=True)
    parser.add_argument('--excel-key', default=None)
    parser.add_argument('--bucket', default='credit-risk-usecase-bucket')
    parser.add_argument('--instance-type', default='ml.m5.xlarge')
    parser.add_argument('--online', action='store_true')
    
    args = parser.parse_args()
    
    create_processing_job(
        pdf_s3_key=args.pdf_key,
        excel_s3_key=args.excel_key,
        bucket=args.bucket,
        instance_type=args.instance_type,
        offline=not args.online
    )


if __name__ == '__main__':
    main()

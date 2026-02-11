# deploy_working_model.py
"""
Deploy a verified working text generation model for the credit risk pipeline.
These models are tested and known to work with SageMaker endpoints.
"""
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import boto3
import json
import time

# Recommended models (in order of preference for this use case)
RECOMMENDED_MODELS = {
    "flan-t5-base": {
        "model_id": "google/flan-t5-base",
        "instance_type": "ml.g5.xlarge",
        "description": "Fast, efficient instruction-following model (250M params)",
        "good_for": "Quick responses, lower cost, good for structured extraction"
    },
    "flan-t5-large": {
        "model_id": "google/flan-t5-large",
        "instance_type": "ml.g5.xlarge",
        "description": "Balanced performance (780M params)",
        "good_for": "Better accuracy, still reasonably fast"
    },
    "flan-t5-xl": {
        "model_id": "google/flan-t5-xl",
        "instance_type": "ml.g5.2xlarge",
        "description": "High accuracy (3B params)",
        "good_for": "Best quality extraction, slower"
    },
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "instance_type": "ml.g5.xlarge",
        "description": "Small but capable LLM (2.7B params)",
        "good_for": "Good reasoning, compact size"
    },
}


def deploy_recommended_model(
    model_choice: str = "flan-t5-base",
    endpoint_name: str = "credit-risk-llm-endpoint",
    region: str = "us-east-2"
):
    """
    Deploy a recommended text generation model.
    
    Args:
        model_choice: One of: flan-t5-base, flan-t5-large, flan-t5-xl, phi-2
        endpoint_name: Name for the endpoint
        region: AWS region
    """
    if model_choice not in RECOMMENDED_MODELS:
        print(f"Error: Unknown model '{model_choice}'")
        print(f"Available models: {', '.join(RECOMMENDED_MODELS.keys())}")
        return
    
    model_info = RECOMMENDED_MODELS[model_choice]
    model_id = model_info["model_id"]
    instance_type = model_info["instance_type"]
    
    print("="*80)
    print(f"Deploying: {model_choice}")
    print("="*80)
    print(f"Model ID: {model_id}")
    print(f"Instance: {instance_type}")
    print(f"Description: {model_info['description']}")
    print(f"Good for: {model_info['good_for']}")
    print("="*80)
    
    # Initialize SageMaker
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create model
    huggingface_model = HuggingFaceModel(
        model_data=None,  # Download from HuggingFace
        role=role,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version='py310',
        env={
            'HF_MODEL_ID': model_id,
            'HF_TASK': 'text2text-generation',  # Important for T5 models
            'MAX_INPUT_LENGTH': '512',
            'MAX_TOTAL_TOKENS': '1024',
        }
    )
    
    # Deploy
    print("\nDeploying endpoint (this will take 5-10 minutes)...")
    print("The model will download from HuggingFace and load into GPU memory.\n")
    
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        container_startup_health_check_timeout=600,
    )
    
    print(f"\n✅ Endpoint '{endpoint_name}' deployed successfully!")
    
    # Wait for stabilization
    print("\nWaiting 30 seconds for endpoint to stabilize...")
    time.sleep(30)
    
    # Test the endpoint
    print("\nTesting endpoint...")
    test_cases = [
        {
            "name": "Simple test",
            "inputs": "What is 2+2?",
            "parameters": {"max_new_tokens": 20}
        },
        {
            "name": "Financial extraction",
            "inputs": "Extract the interest coverage ratio from this text: The company reported an interest coverage ratio of 3.2x for the year.",
            "parameters": {"max_new_tokens": 50}
        },
        {
            "name": "JSON output test",
            "inputs": 'Return a JSON object with the number 42 as the value for key "answer". Return only JSON, no other text.',
            "parameters": {"max_new_tokens": 30}
        }
    ]
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        print(f"Input: {test['inputs'][:80]}...")
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test)
            )
            
            result = json.loads(response['Body'].read().decode())
            print(f"✅ Output: {result}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\nYour endpoint is ready to use!")
    print(f"\nTo use it in your risk pipeline:")
    print(f"""
python risk_pipeline_sagemaker_endpoint.py \\
    --bucket credit-risk-usecase-bucket \\
    --pdf-key input/documents/client_healthy_complex.pdf \\
    --excel-key input/metrics/client_financials_complex.xlsx \\
    --policy-key config/credit_policy.yaml \\
    --endpoint-name {endpoint_name} \\
    --region {region} \\
    --output-prefix output/test-run
""")
    
    return {
        'endpoint_name': endpoint_name,
        'model_id': model_id,
        'instance_type': instance_type
    }


def list_recommended_models():
    """Display all recommended models."""
    print("\n" + "="*80)
    print("RECOMMENDED MODELS FOR CREDIT RISK PIPELINE")
    print("="*80)
    
    for name, info in RECOMMENDED_MODELS.items():
        print(f"\n{name}:")
        print(f"  Model: {info['model_id']}")
        print(f"  Instance: {info['instance_type']}")
        print(f"  Description: {info['description']}")
        print(f"  Good for: {info['good_for']}")
    
    print("\n" + "="*80)
    print("DEPLOYMENT EXAMPLES")
    print("="*80)
    
    print("""
# Best for production (balanced cost/performance):
python deploy_working_model.py deploy --model flan-t5-base

# Best for accuracy:
python deploy_working_model.py deploy --model flan-t5-large

# Best for complex financial reasoning:
python deploy_working_model.py deploy --model phi-2
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy working text generation model")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recommended models')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a model')
    deploy_parser.add_argument(
        '--model',
        choices=list(RECOMMENDED_MODELS.keys()),
        default='flan-t5-base',
        help='Model to deploy'
    )
    deploy_parser.add_argument(
        '--endpoint-name',
        default='credit-risk-llm-endpoint',
        help='Endpoint name'
    )
    deploy_parser.add_argument(
        '--region',
        default='us-east-2',
        help='AWS region'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_recommended_models()
    elif args.command == 'deploy':
        deploy_recommended_model(
            model_choice=args.model,
            endpoint_name=args.endpoint_name,
            region=args.region
        )
    else:
        # Default: show available models
        list_recommended_models()
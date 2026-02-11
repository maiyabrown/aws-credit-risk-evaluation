# diagnose_endpoint.py
"""
Quick diagnostic script to troubleshoot SageMaker endpoint issues.
"""
import boto3
import json
import time
from datetime import datetime, timezone
import sys

def diagnose_endpoint(endpoint_name, region='us-east-2'):
    """Run comprehensive diagnostics on a SageMaker endpoint."""
    
    sm = boto3.client('sagemaker', region_name=region)
    cw_logs = boto3.client('logs', region_name=region)
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    print("="*80)
    print(f"ENDPOINT DIAGNOSTICS: {endpoint_name}")
    print("="*80)
    
    # 1. Check endpoint status
    print("\n[1] ENDPOINT STATUS")
    print("-" * 80)
    try:
        ep = sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"Status: {ep['EndpointStatus']}")
        print(f"Created: {ep['CreationTime']}")
        print(f"Last Modified: {ep['LastModifiedTime']}")
        
        # Check how long since last modification
        now = datetime.now(timezone.utc)
        modified = ep['LastModifiedTime']
        age_minutes = (now - modified).total_seconds() / 60
        
        if age_minutes < 10:
            print(f"\n⚠️  WARNING: Endpoint was modified {age_minutes:.1f} minutes ago")
            print("   The model may still be downloading or loading into GPU memory.")
            print("   Recommendation: Wait 5-10 more minutes before testing.")
        else:
            print(f"\n✅ Endpoint has been stable for {age_minutes:.1f} minutes")
        
        # Get endpoint config
        config_name = ep['EndpointConfigName']
        config = sm.describe_endpoint_config(EndpointConfigName=config_name)
        
        print(f"\nEndpoint Configuration:")
        for variant in config['ProductionVariants']:
            print(f"  Instance Type: {variant['InstanceType']}")
            print(f"  Instance Count: {variant['InitialInstanceCount']}")
            print(f"  Model Name: {variant['ModelName']}")
            
            # Get model details
            model = sm.describe_model(ModelName=variant['ModelName'])
            if 'PrimaryContainer' in model:
                container = model['PrimaryContainer']
                print(f"  Container Image: {container.get('Image', 'N/A')}")
                if 'Environment' in container:
                    print(f"  Environment Variables:")
                    for k, v in container['Environment'].items():
                        print(f"    {k}: {v}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not describe endpoint: {e}")
        return False
    
    # 2. Check CloudWatch logs
    print("\n[2] CLOUDWATCH LOGS (last 20 lines)")
    print("-" * 80)
    try:
        log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        
        # Get most recent log stream
        streams = cw_logs.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )
        
        if streams['logStreams']:
            print(f"Found {len(streams['logStreams'])} log stream(s)")
            
            # Get events from most recent stream
            stream_name = streams['logStreams'][0]['logStreamName']
            print(f"Reading from: {stream_name}\n")
            
            events = cw_logs.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=20,
                startFromHead=False  # Get most recent
            )
            
            # Print last 20 log lines
            for event in events['events']:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                msg = event['message'].strip()
                print(f"[{timestamp}] {msg}")
            
            # Look for key indicators
            all_logs = ' '.join([e['message'] for e in events['events']])
            
            print("\nLog Analysis:")
            if 'loading model' in all_logs.lower():
                print("  ⏳ Model is loading...")
            if 'model loaded' in all_logs.lower():
                print("  ✅ Model loaded successfully")
            if 'cuda out of memory' in all_logs.lower():
                print("  ❌ GPU out of memory - need larger instance")
            if 'error' in all_logs.lower() or 'exception' in all_logs.lower():
                print("  ⚠️  Errors detected in logs")
            if 'timeout' in all_logs.lower():
                print("  ⚠️  Timeout issues detected")
                
        else:
            print("⚠️  No log streams found yet")
            print("   This usually means the container hasn't started")
            
    except cw_logs.exceptions.ResourceNotFoundException:
        print(f"⚠️  Log group not found: {log_group}")
        print("   This is normal for very new endpoints")
    except Exception as e:
        print(f"⚠️  Could not read logs: {e}")
    
    # 3. Test invocation with retries
    print("\n[3] INVOCATION TEST")
    print("-" * 80)
    
    test_payload = {
        "inputs": "Hello",
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 1.0,
            "do_sample": False,
        }
    }
    
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    for attempt in range(3):
        try:
            print(f"\nAttempt {attempt + 1}/3...")
            
            start_time = time.time()
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_payload)
            )
            elapsed = time.time() - start_time
            
            result = json.loads(response['Body'].read().decode())
            
            print(f"✅ SUCCESS (took {elapsed:.2f}s)")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ FAILED: {error_msg}")
            
            # Analyze the error
            if 'timed out' in error_msg.lower():
                print("\n   TIMEOUT ERROR ANALYSIS:")
                print("   - Model is taking >60 seconds to respond")
                print("   - This usually means:")
                if age_minutes < 10:
                    print("     • Model is still loading (MOST LIKELY)")
                    print("     • Recommendation: Wait 10 minutes and try again")
                else:
                    print("     • Model is too slow for this instance type")
                    print("     • Recommendation: Use larger instance or smaller model")
                
            elif 'model error' in error_msg.lower():
                print("\n   MODEL ERROR ANALYSIS:")
                print("   - Check CloudWatch logs above for details")
                print("   - Common causes:")
                print("     • CUDA out of memory")
                print("     • Missing dependencies")
                print("     • Incompatible model format")
            
            if attempt < 2:
                wait_time = 30 * (attempt + 1)
                print(f"\n   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("\n   All attempts failed!")
    
    return False


def suggest_solutions(endpoint_name, region='us-east-2'):
    """Suggest solutions based on diagnostic results."""
    
    print("\n" + "="*80)
    print("RECOMMENDED SOLUTIONS")
    print("="*80)
    
    print("""
1. WAIT AND RETRY (if endpoint is new)
   The model may still be loading. Wait 10 minutes and run:
   
   python diagnose_endpoint.py {endpoint_name}

2. CHECK CLOUDWATCH LOGS IN DETAIL
   Open in browser:
   https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}
   
   Or use CLI:
   aws logs tail /aws/sagemaker/Endpoints/{endpoint_name} --follow --region {region}

3. TRY A SMALLER TEST MODEL
   Delete current endpoint and try a known-working model:
   
   aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {region}
   
   python deploy_llm_endpoint_fixed.py deploy \\
       --model-id "google/flan-t5-base" \\
       --instance-type ml.g5.xlarge \\
       --endpoint-name test-endpoint \\
       --region {region}

4. USE OFFLINE MODE (NO ENDPOINT NEEDED)
   For immediate results, use regex-based extraction:
   
   python risk_pipeline_sagemaker_endpoint.py \\
       --bucket credit-risk-usecase-bucket \\
       --pdf-key input/documents/report.pdf \\
       --policy-key config/credit_policy.yaml \\
       --output-prefix output/test \\
       --offline

5. USE ASYNC INFERENCE
   If model genuinely needs >60s, use async endpoints instead.

6. TRY TEXT GENERATION INFERENCE (TGI) CONTAINER
   More optimized for LLM serving - see TIMEOUT_TROUBLESHOOTING.md
""".format(endpoint_name=endpoint_name, region=region))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        endpoint_name = sys.argv[1]
        region = sys.argv[2] if len(sys.argv) > 2 else 'us-east-2'
    else:
        endpoint_name = 'finance-llm-endpoint'
        region = 'us-east-2'
    
    print(f"\nDiagnosing endpoint: {endpoint_name} in region: {region}\n")
    
    success = diagnose_endpoint(endpoint_name, region)
    
    if not success:
        suggest_solutions(endpoint_name, region)
    else:
        print("\n" + "="*80)
        print("✅ ENDPOINT IS WORKING CORRECTLY")
        print("="*80)
        print("\nYou can now use this endpoint in your risk pipeline:")
        print(f"""
python risk_pipeline_sagemaker_endpoint.py \\
    --bucket credit-risk-usecase-bucket \\
    --pdf-key input/documents/report.pdf \\
    --endpoint-name {endpoint_name} \\
    --region {region} \\
    --output-prefix output/test
""")
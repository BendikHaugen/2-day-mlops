"""
Run batch inference using production model.
"""
import os
import time

import boto3
import sagemaker
from sagemaker import ModelPackage

sm_client = boto3.client('sagemaker', region_name='eu-north-1')
session = sagemaker.Session()
bucket = session.default_bucket()
role = 'arn:aws:iam::305637213530:role/service-role/AmazonSageMaker-ExecutionRole-20251106T100477'


def get_latest_prod_model():
    """Get latest approved model from production."""
    response = sm_client.list_model_packages(
        ModelPackageGroupName='iris-classifier-staging',
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response['ModelPackageSummaryList']:
        raise Exception("No approved models in production registry")
    
    return response['ModelPackageSummaryList'][0]['ModelPackageArn']


def create_test_data():
    """Create sample test data with known iris types."""
    test_data = """5.1,3.5,1.4,0.2
6.2,2.9,4.3,1.3
7.3,2.9,6.3,1.8
4.9,3.0,1.4,0.2
6.4,3.2,4.5,1.5
6.3,3.3,6.0,2.5
5.0,3.6,1.4,0.2
5.9,3.0,5.1,1.8
6.7,3.1,4.4,1.4
7.7,2.8,6.7,2.0"""
    
    # Save to temp file
    temp_file = '/tmp/batch_test_data.csv'
    with open(temp_file, 'w') as f:
        f.write(test_data)
    
    return temp_file


def main():
    print("=" * 60)
    print("Batch Inference with Production Model")
    print("=" * 60)

    # Get production model
    print("\n1️⃣  Getting latest production model...")
    model_arn = get_latest_prod_model()
    print(f"✓ Using model: {model_arn}")

    model_package_details = sm_client.describe_model_package(ModelPackageName=model_arn)
    print(f"\n   Model Package Details:")
    print(f"   - Status: {model_package_details.get('ModelPackageStatus', 'N/A')}")
    print(f"   - Model Approval Status: {model_package_details.get('ModelApprovalStatus', 'N/A')}")
    if 'InferenceSpecification' in model_package_details:
        print(f"   - Has InferenceSpecification: Yes")
        containers = model_package_details['InferenceSpecification'].get('Containers', [])
        print(f"   - Container count: {len(containers)}")
        for i, container in enumerate(containers):
            print(f"      Container {i}: Image={container.get('Image', 'N/A')}, ModelDataUrl={container.get('ModelDataUrl', 'N/A')}")
    else:
        print(f"   - Has InferenceSpecification: No (This might be the issue!)")

    # Create and upload test data
    print("\n2️⃣  Creating test data...")
    test_file = create_test_data()

    print(f"   Uploading to S3...")
    s3 = boto3.client('s3', region_name='eu-north-1')

    timestamp = int(time.time())
    input_key = f'batch-input/test-{timestamp}.csv'
    input_path = f's3://{bucket}/{input_key}'

    s3.upload_file(test_file, bucket, input_key)
    print(f"✓ Test data uploaded to {input_path}")
    
    # Create model package
    print("\n3️⃣  Creating transformer...")
    model_package = ModelPackage(
        role=role,
        model_package_arn=model_arn,
        sagemaker_session=session
    )
    
    output_path = f's3://{bucket}/batch-output/'
    
    transformer = model_package.transformer(
        instance_count=1,
        instance_type='ml.c6i.large',
        output_path=output_path,
        accept='text/csv',
    )
    
    print("✓ Transformer created")
    
    # Run batch transform
    print("\n4️⃣  Starting batch transform job...")
    print("   ⏳ This takes 5-7 minutes (starting instance, loading model, running predictions)...")

    print(f"   Debug: Input path = {input_path}")
    print(f"   Debug: Output path = {output_path}")

    transformer.transform(
        data=input_path,
        content_type='text/csv',
        split_type='Line',
        wait=False,  # Don't block, we'll poll
        logs=True,
    )
    
    # Get job name
    job_name = transformer.latest_transform_job.name
    print(f"   Job name: {job_name}")
    
    # Poll for completion
    print("\n   Waiting for completion...")
    start_time = time.time()

    while True:
        response = sm_client.describe_transform_job(TransformJobName=job_name)
        status = response['TransformJobStatus']

        elapsed = int(time.time() - start_time)
        print(f"   Status: {status} (elapsed: {elapsed}s)", end='\r')

        if status in ['Completed', 'Failed', 'Stopped']:
            print()  # New line
            break

        time.sleep(10)

    full_response = response

    if status == 'Completed':
        print("\n" + "=" * 60)
        print("✅ Batch inference completed successfully!")
        print("=" * 60)
        
        # Download results
        print(f"\n📥 Downloading results...")
        output_key = f"batch-output/{os.path.basename(test_file)}.out"
        output_file = '/tmp/batch_results.csv'
        
        try:
            s3 = boto3.client('s3')
            s3.download_file(bucket, output_key, output_file)
            
            print(f"✓ Results downloaded to: {output_file}")
            print("\n📊 Predictions:")
            print("-" * 60)
            
            # Read and display results
            with open(test_file, 'r') as f:
                inputs = f.readlines()
            
            with open(output_file, 'r') as f:
                predictions = f.readlines()
            
            species = ['Setosa', 'Versicolor', 'Virginica']
            
            for i, (inp, pred) in enumerate(zip(inputs, predictions), 1):
                inp = inp.strip()
                pred = pred.strip()
                species_name = species[int(float(pred))] if pred.replace('.','').isdigit() else pred
                print(f"{i}. Input: {inp}")
                print(f"   Predicted: {species_name} (class {pred})")
                print()
            
            print("-" * 60)
            print(f"\n💾 Full results available at:")
            print(f"   s3://{bucket}/{output_key}")
            
        except Exception as e:
            print(f"⚠️  Could not download results: {e}")
            print(f"   View manually at: s3://{bucket}/batch-output/")
    
    else:
        print(f"\n❌ Job failed with status: {status}")
        print("\n" + "=" * 60)
        print("DEBUG INFO - Transform Job Details:")
        print("=" * 60)

        if 'FailureReason' in full_response:
            print(f"\nFailure Reason: {full_response['FailureReason']}")

        print(f"\nTransform Job Name: {job_name}")
        print(f"Status: {full_response.get('TransformJobStatus', 'N/A')}")

        transform_input = full_response.get('TransformInput', {})
        print(f"TransformInput: {transform_input}")
        s3_uri = transform_input.get('DataSource', {}).get('S3DataSource', {}).get('S3Uri', 'N/A')
        print(f"Input S3 URI: {s3_uri}")

        print(f"Output S3 URI: {full_response.get('TransformOutput', {}).get('S3OutputPath', 'N/A')}")
        print(f"Model Name: {full_response.get('ModelName', 'N/A')}")
        print(f"Instance Count: {full_response.get('TransformResources', {}).get('InstanceCount', 'N/A')}")
        print(f"Instance Type: {full_response.get('TransformResources', {}).get('InstanceType', 'N/A')}")

        if 'TransformStartTime' in full_response and 'TransformEndTime' in full_response:
            duration = (full_response['TransformEndTime'] - full_response['TransformStartTime']).total_seconds()
            print(f"Duration: {duration:.1f} seconds")

        print("\nTo debug further:")
        print(f"1. Check CloudWatch logs in AWS console:")
        print(f"   https://console.aws.amazon.com/cloudwatch/home?region=eu-north-1#logStream:")
        print(f"2. Check Transform Job in SageMaker console:")
        print(f"   https://console.aws.amazon.com/sagemaker/home?region=eu-north-1#/transform-jobs/{job_name}")
        print("=" * 60)


if __name__ == "__main__":
    main()
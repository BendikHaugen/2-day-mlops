import logging
import os

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_pipeline(
    region,
    role,
    training_image_uri,
    evaluation_image_uri,
    model_group_name="iris-classifier-staging",
    pipeline_name="IrisPipeline",
):
    """Create the Iris classifier pipeline."""
    
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    # Parameters
    model_group_param = ParameterString(
        name="ModelGroupName",
        default_value=model_group_name
    )
    
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    # --- Training Step ---
    training_estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session,
        environment={
            "TRAINING_IMAGE": training_image_uri,
        }
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=training_estimator,
        cache_config=cache_config,
    )
    
    # --- Evaluation Step ---
    evaluation_processor = Processor(
        image_uri=evaluation_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.t3.medium",
        sagemaker_session=sagemaker_session,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
            ),
        ],
        property_files=[evaluation_report],
        cache_config=cache_config,
    )
    
    # --- Register Model Step ---
    model = Model(
        image_uri=training_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            content_type="application/json"
        )
    )
    
    register_step = RegisterModel(
        name="RegisterModel",
        model=model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.c6i.large"],
        model_package_group_name=model_group_param,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[model_group_param],
        steps=[training_step, evaluation_step, register_step],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline


def main():
    """Deploy the pipeline."""
    
    # Configuration
    region = os.environ.get("AWS_REGION", "eu-north-1")
    
    # Get account ID
    sts_client = boto3.client('sts', region_name=region)
    account_id = sts_client.get_caller_identity()['Account']
    
    # IAM Role
    role = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20251106T100477"
    
    # Docker image URIs
    training_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/iris-classifier-training:latest"
    evaluation_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/iris-classifier-evaluation:latest"
    
    log.info("=" * 60)
    log.info("Creating SageMaker Pipeline")
    log.info("=" * 60)
    log.info(f"Region: {region}")
    log.info(f"Account: {account_id}")
    log.info(f"Training image: {training_image}")
    log.info(f"Evaluation image: {evaluation_image}")
    log.info("=" * 60)
    
    # Create pipeline
    pipeline = get_pipeline(
        region=region,
        role=role,
        training_image_uri=training_image,
        evaluation_image_uri=evaluation_image,
    )
    
    # Deploy pipeline
    log.info("\nUpserting pipeline...")
    pipeline.upsert(role_arn=role)
    log.info("✓ Pipeline deployed successfully!")
    
    # Optionally start execution
    if os.environ.get("START_EXECUTION"):
        log.info("\nStarting pipeline execution...")
        execution = pipeline.start()
        log.info(f"✓ Execution started: {execution.arn}")
        log.info("\nView in console:")
        log.info(f"https://console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/IrisPipeline/executions")
    else:
        log.info("\n" + "=" * 60)
        log.info("Pipeline deployed but not started")
        log.info("To start execution:")
        log.info("  Option 1: START_EXECUTION=1 python mlops/pipelines/iris_pipeline.py")
        log.info("  Option 2: AWS Console → SageMaker → Pipelines → IrisPipeline → Create execution")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
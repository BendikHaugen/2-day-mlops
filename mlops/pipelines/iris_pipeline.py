import logging
import os

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
    """
    Creates a SageMaker Pipeline using custom Docker images.

    Args:
        region: AWS region
        role: SageMaker execution role ARN
        training_image_uri: ECR URI for training image
        evaluation_image_uri: ECR URI for evaluation image
        model_group_name: Model Registry group name
        pipeline_name: Name of the pipeline

    Returns:
        Pipeline object
    """
    sagemaker_session = sagemaker.Session()

    # Parameters
    model_group_param = ParameterString(
        name="ModelGroupName",
        default_value=model_group_name
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    default_instance_type = "ml.m5.large"

    # --- Training Step ---
    estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type=default_instance_type,
        sagemaker_session=sagemaker_session,
        environment={
            "TRAINING_IMAGE": training_image_uri,
        }
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        cache_config=cache_config,
    )

    # --- Evaluation Step ---
    evaluation_processor = Processor(
        image_uri=evaluation_image_uri,
        role=role,
        instance_count=1,
        instance_type=default_instance_type,
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
        inference_instances=[default_instance_type],
        transform_instances=[default_instance_type],
        model_package_group_name=model_group_param,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # --- Create Pipeline ---
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[model_group_param],
        steps=[training_step, evaluation_step, register_step],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def main():
    """Main function to create/update and optionally start the pipeline."""

    # Configuration
    region = os.environ.get("AWS_REGION", "eu-north-1")
    role = os.environ.get(
        "SAGEMAKER_ROLE",
        "arn:aws:iam::305637213530:role/service-role/AmazonSageMaker-ExecutionRole-20251106T100477"
    )

    # Docker image URIs (passed from CI/CD or defaults)
    training_image_uri = os.environ.get(
        "TRAINING_IMAGE_URI",
        "305637213530.dkr.ecr.eu-north-1.amazonaws.com/iris-classifier-training:latest"
    )
    evaluation_image_uri = os.environ.get(
        "EVALUATION_IMAGE_URI",
        "305637213530.dkr.ecr.eu-north-1.amazonaws.com/iris-classifier-evaluation:latest"
    )

    log.info(f"Creating pipeline with images:")
    log.info(f"  Training: {training_image_uri}")
    log.info(f"  Evaluation: {evaluation_image_uri}")

    # Create pipeline
    pipeline = get_pipeline(
        region=region,
        role=role,
        training_image_uri=training_image_uri,
        evaluation_image_uri=evaluation_image_uri,
    )

    # Upsert pipeline
    log.info("Upserting pipeline definition...")
    pipeline.upsert(role_arn=role)
    log.info("✓ Pipeline upsert complete")

    # Optionally start execution (for CI/CD)
    if os.environ.get("CI_RUN"):
        log.info("CI_RUN detected. Starting pipeline execution...")
        execution = pipeline.start()
        log.info(f"✓ Started execution: {execution.arn}")

        # Wait for completion (optional)
        if os.environ.get("WAIT_FOR_COMPLETION"):
            log.info("Waiting for pipeline to complete...")
            execution.wait(delay=30, max_attempts=60)
            status = execution.describe()['PipelineExecutionStatus']

            if status != 'Succeeded':
                log.error(f"Pipeline failed with status: {status}")
                raise Exception("Pipeline execution failed")

            log.info("✓ Pipeline execution succeeded!")
    else:
        log.info("Pipeline ready. Start from SageMaker Studio UI or set CI_RUN=1")


if __name__ == "__main__":
    main()

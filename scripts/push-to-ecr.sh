#!/bin/bash
set -e

echo "Push Docker Images to ECR"
echo "======================================"
echo ""

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=eu-north-1

echo "Account: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo ""


echo "🔐 Step 1: Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "✓ Logged in to ECR"
echo ""

# Step 2: Build AMD64 images
echo "🏗️  Step 2: Building AMD64 images (for AWS)..."
echo ""

echo "Building training image..."
docker build --platform linux/amd64 \
  -f docker/training/Dockerfile \
  -t iris-training:aws \
  .

echo ""
echo "Building evaluation image..."
docker build --platform linux/amd64 \
  -f docker/evaluation/Dockerfile \
  -t iris-evaluation:aws \
  .

echo ""
echo "✓ Images built for AMD64"
echo ""

echo "🔍 Step 3: Verifying architecture..."
TRAINING_ARCH=$(docker inspect iris-training:aws --format '{{.Architecture}}')
EVAL_ARCH=$(docker inspect iris-evaluation:aws --format '{{.Architecture}}')

if [ "$TRAINING_ARCH" != "amd64" ]; then
    echo "❌ Training image is $TRAINING_ARCH, should be amd64!"
    exit 1
fi

if [ "$EVAL_ARCH" != "amd64" ]; then
    echo "❌ Evaluation image is $EVAL_ARCH, should be amd64!"
    exit 1
fi

echo "✓ Training image: $TRAINING_ARCH"
echo "✓ Evaluation image: $EVAL_ARCH"
echo ""

# Step 4: Tag images
echo "🏷️  Step 4: Tagging images for ECR..."
TRAINING_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/iris-classifier-training"
EVAL_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/iris-classifier-evaluation"

# Tag with version and latest
VERSION="v1.0.0"

docker tag iris-training:aws $TRAINING_URI:$VERSION
docker tag iris-training:aws $TRAINING_URI:latest

docker tag iris-evaluation:aws $EVAL_URI:$VERSION
docker tag iris-evaluation:aws $EVAL_URI:latest

echo "✓ Tagged images"
echo ""

# Step 5: Push images
echo "📤 Step 5: Pushing images to ECR..."
echo ""

echo "Pushing training image..."
docker push $TRAINING_URI:$VERSION
docker push $TRAINING_URI:latest

echo ""
echo "Pushing evaluation image..."
docker push $EVAL_URI:$VERSION
docker push $EVAL_URI:latest

echo ""
echo "✓ Images pushed to ECR"
echo ""

# Step 6: Verify in ECR
echo "🔍 Step 6: Verifying images in ECR..."
echo ""

echo "Training images:"
aws ecr describe-images \
  --repository-name iris-classifier-training \
  --region $AWS_REGION \
  --query 'imageDetails[*].[imageTags[0],imagePushedAt]' \
  --output table

echo ""
echo "Evaluation images:"
aws ecr describe-images \
  --repository-name iris-classifier-evaluation \
  --region $AWS_REGION \
  --query 'imageDetails[*].[imageTags[0],imagePushedAt]' \
  --output table

echo ""
echo "======================================"
echo "🎉 Phase 2 Complete!"
echo "======================================"
echo ""
echo "Images available at:"
echo "  Training:   $TRAINING_URI:latest"
echo "  Evaluation: $EVAL_URI:latest"
echo ""
echo "Next: Phase 3 - Create SageMaker Pipeline"
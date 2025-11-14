# Docker OCI Manifest Issue with AWS SageMaker

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [For Software Engineers](#for-software-engineers)
3. [Explain Like I'm 5 (ELI5)](#explain-like-im-5-eli5)

---

## Executive Summary

When building Docker images on macOS and pushing them to AWS ECR, the images end up in **OCI Image Manifest format** (`application/vnd.oci.image.manifest.v1+json`). AWS SageMaker only accepts **Docker Manifest V2 format** (`application/vnd.docker.distribution.manifest.v2+json`). This causes SageMaker API calls to fail with a `ValidationException`.

**Root Cause:** macOS Docker Desktop uses a containerd backend that defaults to OCI format for all images, regardless of how they're built.

**Key Insight:** This validation is **not a technical requirement**—it's an AWS policy decision. The underlying container runtime (containerd/Docker daemon) can execute both OCI and Docker Manifest V2 formats perfectly fine. Training and Processing jobs work with OCI images without issues, but the `CreateModel` API has an artificial validation layer that rejects them.

**Solution:** Build Docker images on a Linux system where Docker uses the Docker Manifest V2 format by default.

**Best Practice:** Use GitHub Actions or CI/CD pipeline (not local macOS builds) for anything that needs `CreateModel` compatibility. Also consider advocating to AWS for OCI format support.

---

## For Software Engineers

### The Problem

#### What is a Docker Manifest?

A Docker manifest is metadata that describes a Docker image. It includes:
- The image's schema version
- References to image layers (compressed filesystem diffs)
- Image configuration (environment variables, entry points, etc.)
- Media type indicating the format specification

When you push an image to a registry like ECR, the manifest is what the registry stores to describe that image.

#### Docker Manifest V2 vs OCI Image Manifest

There are two competing standards for describing container images:

1. **Docker Manifest V2** (`application/vnd.docker.distribution.manifest.v2+json`)
   - The standard Docker format
   - Defined by Docker Inc.
   - Widely supported, including by AWS SageMaker

2. **OCI Image Manifest** (`application/vnd.oci.image.manifest.v1+json`)
   - The Open Container Initiative standard
   - Designed to be a universal standard
   - Technically "better" but not universally supported
   - AWS SageMaker does NOT support this format

#### Why macOS Produces OCI Format

macOS Docker Desktop (version 4.1+) uses **containerd** as its runtime backend instead of the native Docker daemon. Containerd's default behavior when building and storing images is to use OCI format.

When you run:
```bash
docker build -f Dockerfile -t my-image:latest .
```

On macOS, Docker Desktop:
1. Forwards the build to containerd
2. containerd builds the image and stores it in OCI format
3. When you push with `docker push`, the OCI manifest is sent to ECR
4. ECR stores the image with the OCI manifest metadata

#### The SageMaker Validation

Only **certain SageMaker APIs** validate the image manifest format strictly. This is key to understanding why training and evaluation images don't fail, but inference does.

#### APIs That DON'T Validate Manifest Format
- **SageMaker Training API** (`CreateTrainingJob`)
  - Used by: `TrainingStep` with `Estimator`
  - Accepts: Any image format (OCI or Docker)
  - Why: The training container just needs to be runnable; manifest format doesn't matter

- **SageMaker Processing API** (`CreateProcessingJob`)
  - Used by: `ProcessingStep` with `Processor`
  - Accepts: Any image format (OCI or Docker)
  - Why: The processing container just needs to be runnable; manifest format doesn't matter

#### APIs That DO Validate Manifest Format
- **SageMaker CreateModel API** (`CreateModel`)
  - Used by: `RegisterModel` step, Batch Transform, Inference endpoints
  - Accepts: Docker Manifest V2 ONLY
  - Rejects: OCI formats (both index and manifest)
  - Why: **Arbitrary validation requirement** (not technically necessary)

#### Why This Validation Exists (But Shouldn't)

The `CreateModel` API validates manifest format as a **compatibility guarantee**, not a technical requirement:

```
CreateTrainingJob:
  Image (OCI format) → containerd runtime → Runs fine ✓
  (No validation, just executes)

CreateModel:
  Image (OCI format) → AWS API validation → REJECTED ✗
  (Validation layer blocks it BEFORE execution)
  Image (OCI format) → Would run fine if allowed → Would work ✓
  (containerd runtime could execute it)
```

**The reality:** If the validation layer were removed, OCI format images would work perfectly fine because the underlying container runtime (containerd/Docker daemon) supports both formats equally well.

The validation is essentially AWS saying: "We only officially support Docker Manifest V2 for model serving." This is a **policy decision**, not a technical limitation.

This means:
```python
# Training - WORKS with OCI
model_from_training = Model(image_uri="...", ...)
training_job = model_from_training.fit()  # ✓ OK

# CreateModel - FAILS with OCI
model = Model(image_uri="...", ...)
model.create()  # ✗ ValidationException!

# Batch Transform - FAILS with OCI
transformer = model_package.transformer(...)
transformer.transform(...)  # ✗ ValidationException!
```

The error occurs in SageMaker's `CreateModel` API when it tries to fetch the image manifest from ECR and validates that it's Docker Manifest V2 format. If it finds OCI format, it rejects it.

### Identifying the Issue

#### Error Message

```
ClientError: An error occurred (ValidationException) when calling the CreateModel operation:
Unsupported manifest media type application/vnd.oci.image.manifest.v1+json for image
305637213530.dkr.ecr.eu-north-1.amazonaws.com/iris-classifier-inference@sha256:abc123...
Ensure that valid manifest media type is used for specified image.
```

Key indicators:
- `ValidationException` from CreateModel
- **"Unsupported manifest media type"** - this is the smoking gun
- **"application/vnd.oci.image"** - confirms it's OCI format

#### Checking Manifest Format Manually

You can verify an image's manifest format using AWS CLI:

```bash
aws ecr batch-get-image \
  --repository-name iris-classifier-inference \
  --image-ids imageTag=latest \
  --region eu-north-1 \
  --query 'images[0].imageManifestMediaType' \
  --output text
```

Output for problematic images:
```
application/vnd.oci.image.manifest.v1+json
```

Output for correct images:
```
application/vnd.docker.distribution.manifest.v2+json
```

Or view the full manifest:
```bash
aws ecr batch-get-image \
  --repository-name iris-classifier-inference \
  --image-ids imageTag=latest \
  --region eu-north-1 \
  --query 'images[0].imageManifest' \
  --output json | python3 -m json.tool
```

Look at the top-level `"mediaType"` field.

### Why `--platform linux/amd64` Doesn't Help

You might think specifying `--platform linux/amd64` would help:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile \
  -t my-image:latest \
  .
```

It doesn't because the `--platform` flag tells Docker which platform to build for, but containerd still stores the result in OCI format. The flag controls the BUILD target, not the MANIFEST format.

### Root Cause Analysis

```
macOS Docker Desktop
    ↓
Uses containerd backend
    ↓
containerd defaults to OCI format for storage
    ↓
`docker push` sends OCI manifest to ECR
    ↓
ECR stores image with OCI manifest metadata
    ↓
SageMaker API rejects OCI manifests
    ↓
CreateModel fails
```

### The Solution: Build on Linux

When Docker runs on Linux, it uses the native Docker daemon (not containerd), which by default produces Docker Manifest V2 format:

```
Linux Docker (native daemon)
    ↓
Builds image in Docker Manifest V2 format
    ↓
`docker push` sends Docker Manifest V2 to ECR
    ↓
ECR stores image with Docker Manifest V2 metadata
    ↓
SageMaker API accepts it
    ↓
CreateModel succeeds
```

#### Implementation Options

**Option 1: AWS EC2 Instance (Simplest)**
1. Launch t3.small Amazon Linux 2 instance
2. SSH in: `ssh -i key.pem ec2-user@ip`
3. Install Docker: `yum install docker && systemctl start docker`
4. Clone repo and run: `bash scripts/push-to-ecr.sh`
5. Terminate instance

**Option 2: GitHub Actions (For CI/CD)**
Create `.github/workflows/docker-build.yml`:
```yaml
name: Build Docker Images
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: docker/setup-buildx-action@v1
      - uses: docker/login-action@v1
        with:
          registry: ${{ secrets.AWS_REGISTRY }}
          username: AWS
          password: ${{ secrets.AWS_ECR_PASSWORD }}
      - uses: docker/build-push-action@v2
        with:
          file: docker/inference/Dockerfile
          push: true
          tags: ${{ secrets.AWS_REGISTRY }}/iris-classifier-inference:latest
```

**Option 3: Docker Desktop BuildKit on macOS (Workaround)**
Set environment variable and use `docker buildx`:
```bash
docker buildx build \
  --platform linux/amd64 \
  --push \
  -f Dockerfile \
  -t my-image:latest \
  .
```

Note: This may still produce OCI format on macOS depending on buildx configuration.

**Option 4: Use Docker BuildKit on macOS (More Complex)**
Configure Docker to use containerd v1.5+ which has better manifest compatibility, but this is unreliable.

### Best Practice Solutions

#### Option 1: Proper MLOps Workflow (RECOMMENDED - Industry Best Practice)
**Build Docker images via CI/CD pipeline (GitHub Actions, Jenkins, GitLab CI, etc.) on Linux:**

**Implementation:**
1. Add Docker build steps to your CI/CD pipeline (e.g., `.github/workflows/main.yml`)
2. Run builds on Linux runners (e.g., `ubuntu-latest`)
3. Images automatically push to ECR with correct Docker Manifest V2 format
4. SageMaker pipeline uses the pre-built images from ECR

**Example GitHub Actions workflow:**
```yaml
jobs:
  build-and-push:
    runs-on: ubuntu-latest  # ← Linux runner produces Docker Manifest V2
    steps:
      - uses: actions/checkout@v3
      - uses: aws-actions/configure-aws-credentials@v2
      - name: Build and push inference image
        run: |
          docker build --platform linux/amd64 \
            -f docker/inference/Dockerfile \
            -t $ECR_REGISTRY/iris-classifier-inference:latest \
            .
          docker push $ECR_REGISTRY/iris-classifier-inference:latest
```

**Pros:**
- Industry standard MLOps practice
- Guarantees Docker Manifest V2 format
- Automated and reproducible
- Scales to multiple images and environments
- Prevents "works on my machine" issues
- This is how production ML systems work

**Cons:**
- Requires GitHub Actions setup (minimal)
- Initial configuration time (30 minutes)

#### Option 2: Advocate for Change (Long-term)
**Report this as a limitation to AWS:**
1. Open an AWS Support case requesting OCI format support for `CreateModel` API
2. Reference that:
   - Training and Processing APIs accept OCI format without issues
   - OCI is the industry standard (CNCF/OCI)
   - macOS Docker Desktop now defaults to OCI format
   - The validation is artificial (runtime supports both formats)

**AWS should:**
- Remove or relax the manifest format validation
- Or at minimum, support OCI format alongside Docker Manifest V2

**Pros:**
- Solves the problem at the root
- Benefits all AWS customers on macOS
- Aligns with industry standards

**Cons:**
- Requires AWS implementation time
- May take months/years for change

#### Option 3: Hybrid Approach (Practical)
**Short-term:** Build on Linux (use GitHub Actions)
**Long-term:** File AWS feature request for OCI support

This way:
- You can ship your project now
- You contribute to improving AWS for the community
- Future projects may benefit from AWS's fix

### Verification

After building on Linux and pushing to ECR, verify:

```bash
aws ecr batch-get-image \
  --repository-name iris-classifier-inference \
  --image-ids imageTag=latest \
  --region eu-north-1 \
  --query 'images[0].imageManifestMediaType' \
  --output text
```

Should output:
```
application/vnd.docker.distribution.manifest.v2+json
```

Then batch inference will work:
```bash
python scripts/run_batch_inference.py
```

### Why This Matters

- **Container Registry Standard:** The Docker Manifest V2 format has been the Docker standard for years and is the format most container orchestration systems expect
- **OCI as Universal Standard:** OCI is attempting to be a universal standard, but adoption is still incomplete in AWS services
- **SageMaker Design Decision:** SageMaker's validation was likely designed to ensure compatibility with specific runtime requirements
- **Platform Differences:** This issue highlights how the same Dockerfile can produce different outputs on different operating systems

---

## Explain Like I'm 5 (ELI5)

### The Basic Problem

Imagine you're sending a package to a friend, and you need to include an instruction manual that says "how to put this together."

- **Docker Manifest** = the instruction manual
- **Image in ECR** = the package

There are two different instruction manual formats:
1. **Docker Manifest V2** - The standard format everyone uses (like a yellow manual)
2. **OCI Manifest** - A newer, fancier format (like a purple manual)

When you build on **Mac**, it creates packages with **purple manuals**.
When you build on **Linux**, it creates packages with **yellow manuals**.

AWS SageMaker says: "I only accept **yellow manuals**! I don't know what to do with purple ones!"

### Why It Only Fails for Inference (Not Training/Evaluation)

Here's a surprising twist:

- **Training image** = used to train the model (purple manual = OK! ✓)
- **Evaluation image** = used to evaluate the model (purple manual = OK! ✓)
- **Inference image** = used to serve predictions (purple manual = NOT OK! ✗)

Why the difference?

- **Training and Evaluation jobs**: "Just run this container, I don't care about the manual format"
- **Inference/Model Serving**: "This container needs to work with our strict inference infrastructure, manual MUST be yellow!"

It's like:
- **Training job**: "Hey truck, here's a box to move. It has instructions, but I don't care what format they're in."
- **Inference job**: "Hey forklift, here's a box to load onto the truck. But the forklift is picky—it ONLY accepts yellow instruction manuals!"

### Why It Happens

Your Mac has a helper program called "containerd" that packs your Docker images. Containerd's favorite way to pack things is with **purple manuals**.

Linux computers have a different way of packing called the "native Docker daemon." It likes **yellow manuals**.

### How to Fix It

You need to get the package made on a Linux computer (which will use the yellow manual):

**Option 1: Rent a Linux Computer (EC2)**
- Go to AWS and rent a little Linux computer for an hour
- Tell it to build your Docker image
- It will create yellow manuals
- Stop renting it (don't waste money!)

**Option 2: Use GitHub's Computers**
- Tell GitHub "whenever I push code, please build my Docker images"
- GitHub has Linux computers in the cloud
- They'll create yellow manuals automatically

**Option 3: Ask Your Team**
- If someone at work has a Linux laptop, they can build it
- Linux creates yellow manuals

### The Lesson

- **Mac creates purple manuals** (OCI format)
- **Linux creates yellow manuals** (Docker format)
- **SageMaker only likes yellow manuals**
- **So build on Linux!**

### Real World Analogy

Think of it like international recipes:
- American recipe (Docker format) - uses cups, tablespoons, Fahrenheit
- Metric recipe (OCI format) - uses grams, liters, Celsius

Your kitchen scale only works with American recipes. Even though metric recipes work great in other kitchens, YOUR scale doesn't understand them.

Solution: Use American recipes in your kitchen, or buy a new scale.

Similarly: Use Docker format in SageMaker, or request OCI format support from AWS.

---

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Problem** | macOS Docker builds produce OCI manifest format |
| **Symptom** | `ValidationException: Unsupported manifest media type application/vnd.oci.image.manifest.v1+json` |
| **Check** | `aws ecr batch-get-image ... --query 'images[0].imageManifestMediaType'` |
| **Root Cause** | macOS Docker Desktop uses containerd which defaults to OCI format |
| **Solution** | Build images on Linux using native Docker daemon |
| **Best Option** | AWS EC2 t3.small instance running Amazon Linux 2 |
| **Easiest CI/CD** | GitHub Actions with ubuntu-latest runner |

## Why Only Inference Fails (Not Training/Evaluation)

| SageMaker Component | Uses API | Manifest Validation | Status |
|---|---|---|---|
| **Training** | `CreateTrainingJob` | Lenient (any format) | ✓ Works |
| **Evaluation** | `CreateProcessingJob` | Lenient (any format) | ✓ Works |
| **Inference/Batch** | `CreateModel` | Strict (Docker V2 only) | ✗ Fails |

**Key Insight:** Only `CreateModel` API validates manifest format. Training and Processing APIs don't care because they just need to execute the container. Model serving is strict because it requires Docker Manifest V2 compatibility.

---

## References

### AWS SageMaker APIs (Official Documentation)
- [CreateModel API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html) - **The strict API that validates manifest format**
- [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) - Lenient, accepts any image format
- [CreateProcessingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html) - Lenient, accepts any image format
- [Docker Container Images for Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-containers-inference.html) - Official guide on inference containers
- [Docker Container Images for Training](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-problem-solving.html) - Official guide on training containers

### Container Standards & Formats
- [Docker Image Manifest V2 Spec](https://docs.docker.com/registry/spec/manifest-v2-2/) - The Docker standard that SageMaker expects
- [OCI Image Spec](https://github.com/opencontainers/image-spec/blob/main/manifest.md) - The newer universal standard that SageMaker doesn't support
- [containerd Documentation](https://containerd.io/) - The container runtime macOS Docker Desktop uses
- [macOS Docker Desktop Release Notes](https://docs.docker.com/desktop/release-notes/) - Details on why macOS uses containerd

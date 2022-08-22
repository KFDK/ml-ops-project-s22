PROJECT_ID=better-mldtu
BUCKET_NAME=better-mldtu-aiplatform
REGION=europe-west1
IMAGE_REPO_NAME=mlops_pytorch_custom_container
IMAGE_TAG=mlops_pytorch_gpu
IMAGE_URI=gcr.io/better-mldtu/mlops_pytorch_custom_container:mlops_pytorch_gpu
MODEL_DIR=pytorch_model_$(date +%Y%m%d_%H%M%S)
JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --scale-tier BASIC_GPU \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    -- \
    --model-dir="gs://$BUCKET_NAME/$MODEL_DIR"


gcloud ai-platform jobs stream-logs ${JOB_NAME}
from google.cloud import aiplatform

VERSION = 1
APP_NAME = "model"
model_display_name = f"{APP_NAME}-v{VERSION}"
model_description = "PyTorch based text classifier with custom container"

MODEL_NAME = APP_NAME
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [7080]

IMG_NAME = "test_img"
IMG_TAG = "gcr.io/better-mldtu/mlops_pytorch_custom_container:" + IMG_NAME

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=IMG_TAG,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)

model.wait()

print(model.display_name)
print(model.resource_name)


endpoint_display_name = f"{APP_NAME}-endpoint"
endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)


traffic_percentage = 100
machine_type = "n1-standard-4"
deployed_model_display_name = model_display_name
min_replica_count = 1
max_replica_count = 3
sync = True

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=deployed_model_display_name,
    machine_type=machine_type,
    traffic_percentage=traffic_percentage,
    sync=sync,
)

model.wait()

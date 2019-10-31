source ~/data/setup_env.sh

python3 main.py -i cam -m0 ./openvino_models/person-detection/FP32/pedestrian-detection-adas-0002.xml -m1 ./openvino_models/person-analyze/FP32/person-attributes-recognition-crossroad-0200.xml -m2 ./openvino_models/face-detection/FP32/face-detection-adas-0001.xml -m3 ./openvino_models/face-analyze/FP32/face.xml -m4 ./openvino_models/fire-detection/FP32/firenet.xml -d CPU -l /opt/intel/2019_r1/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

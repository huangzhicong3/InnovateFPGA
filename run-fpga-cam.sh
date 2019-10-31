source ~/data/setup_env.sh

python3 main.py -i cam -m0 ./openvino_models/person-detection/FP16/pedestrian-detection-adas-0002.xml -m1 ./openvino_models/person-analyze/FP16/person-attributes-recognition-crossroad-0200.xml -m2 ./openvino_models/face-detection/FP16/face-detection-adas-0001.xml -m3 ./openvino_models/face-analyze/FP16/face.xml -m4 ./openvino_models/fire-detection/FP16/firenet.xml -d HETERO:FPGA,CPU -l /opt/intel/2019_r1/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so

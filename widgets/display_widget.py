from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2

from constants import *
from debug.debug import debug
from argparse import ArgumentParser, SUPPRESS

import os
import os.path as osp
import sys

import time
import numpy as np
import logging as log
import xlwt as wt
import xlrd as rd
from xlutils.copy import copy

from openvino.inference_engine import IENetwork, IEPlugin



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m0", "--person_model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-m1", "--attribute_model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-m2", "--face-dete_model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-m3", "--face_anyl_model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-m4", "--fire_model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-i", "--input", help="Required. cam means read camera or you can input a path to a video.",required=True,type=str)
    args.add_argument("-iv","--in_v_num",help="Optional. Num of frame in vertical direction.", required=False,
                      type=int,default=1)
    args.add_argument("-ih", "--in_h_num", help="Optional. Num of frame in horizontal direction.", required=False,
                      type=int, default=1)
    args.add_argument("-l", "--cpu_extension",help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--data", help="Optional. Path to a face-id mapping file", default='user_id.xls', type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=True,
                      action="store_true")
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    return parser

def double_init(plugin, dete_net, anyl_net):
    '''          4. prepare input blobs            '''
    log.info("Preparing input blobs")
    fd_input_blob = next(iter(dete_net.inputs))
    fa_input_blob = next(iter(anyl_net.inputs))
    n, c, h, w = dete_net.inputs[fd_input_blob].shape
    an, ac, ah, aw = anyl_net.inputs[fa_input_blob].shape

    '''         5. load network to the plugin       '''
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=dete_net, num_requests=2)  # for asyn , num_requests = 2
    lm_exec_net = plugin.load(network=anyl_net)
    del dete_net
    del anyl_net
    return exec_net, lm_exec_net, n, c, h, w, an, ac, ah, aw


def single_init(plugin, dete_net):
    '''          4. prepare input blobs            '''
    log.info("Preparing input blobs")
    fd_input_blob = next(iter(dete_net.inputs))
    n, c, h, w = dete_net.inputs[fd_input_blob].shape

    '''         5. load network to the plugin       '''
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=dete_net, num_requests=2)  # for asyn , num_requests = 2
    del dete_net
    return exec_net, n, c, h, w


def fire_dete(exec_net, n, c, h, w,
              frame, next_frame, cur_request_id, next_request_id):
    (c_h, c_w, _) = frame.shape
    pic = np.zeros((1, h, w, 3))
    # divide the image into 6 parts -- cost 2.7ms
    pic[0] = cv2.resize(next_frame[:, :, :], (w, h))
    #pic[0] = cv2.resize(next_frame[0:int(c_h / 2), 0:int(c_w / 3), :], (w, h))
    #pic[1] = cv2.resize(next_frame[0:int(c_h / 2), int(c_w / 3):int(c_w * 2 / 3), :], (w, h))
    #pic[2] = cv2.resize(next_frame[0:int(c_h / 2), int(c_w * 2 / 3):c_w, :], (w, h))
    #pic[3] = cv2.resize(next_frame[int(c_h / 2):c_h, 0:int(c_w / 3), :], (w, h))
    #pic[4] = cv2.resize(next_frame[int(c_h / 2):c_h, int(c_w / 3):int(c_w * 2 / 3), :], (w, h))
    #pic[5] = cv2.resize(next_frame[int(c_h / 2):c_h, int(c_w * 2 / 3):int(c_w), :], (w, h))
    pic = pic.transpose((0, 3, 1, 2))

    exec_net.start_async(request_id=next_request_id, inputs={"input": pic})
    if exec_net.requests[cur_request_id].wait(-1) == 0:
       
        res = exec_net.requests[cur_request_id].outputs["InceptionV1/Logits/conv_out/Conv2D"]
        #print(res[0])
        if (res[0] >= 62) | (res[0] <= 35):
            cv2.putText(frame, "FIRE", (int(c_w / 16), int(c_h / 4)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "SAFE", (int(c_w / 16), int(c_h / 4)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
        #for i in range(6):
        #   if (res[i] >= 80) | (res[i] <= 20):
        #       fire = i
        #        break
        #    else:
        #        fire = 6
        #if fire == 6:
        #    cv2.putText(frame, "SAFE", (int(c_w / 16), int(c_h / 4)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
        #else:
        #    cv2.rectangle(frame, (int(c_w / 3) * (fire % 3), int(c_h / 2) * (fire // 3)),
        #                  (int(c_w / 3) * ((fire % 3) + 1), int(c_h / 2) * ((fire // 3) + 1)), (0, 0, 255), 10)
        #    cv2.putText(frame, "FIRE", (int(c_w / 16), int(c_h / 4)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
    return frame


def face_dete(exec_net, lm_exec_net, n, c, h, w, an, ac, ah, aw, \
              frame, next_frame, cur_request_id, next_request_id, user,id):
    '''         6. process           '''
    attr_res = []
    (initial_h, initial_w, _) = frame.shape
    in_frame = cv2.resize(next_frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    exec_net.start_async(request_id=next_request_id, inputs={"data": in_frame})  #

    if exec_net.requests[cur_request_id].wait(-1) == 0:  # wait for done signal
        end1 = time.time()  # store the time
        # det_time = inf_end - inf_start
        res = exec_net.requests[cur_request_id].outputs["detection_out"]  # get the result
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > 0.5:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                              2)  # Draw a box around the person detected
                if xmin > 0 and ymin > 0 and (xmax < initial_w) and (ymax < initial_h):
                    roi = frame[ymin:ymax, xmin:xmax, :]
                    face_roi = cv2.resize(roi, (aw, ah))
                    face_roi = face_roi.transpose((2, 0, 1))
                    face_roi = face_roi.reshape((an, ac, ah, aw))

                    lm_exec_net.infer(
                        inputs={"batch_join/fifo_queue": face_roi})  # start infer the person roi and wait for done
                    attr_res = lm_exec_net.requests[0].outputs["InceptionResnetV1/Bottleneck/MatMul"]

                    '''                compare the result vector with all user               '''
                    show = "unknown"
                    min_dist = 100
                    identity = 0

                    for o in range(user.__len__()):
                        dist = np.linalg.norm(attr_res - user[o])

                        if dist < min_dist:
                            min_dist = dist
                            identity = o
                    if min_dist < 20:
                        show = id[identity]
                    cv2.putText(frame, show, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    return frame, attr_res

def person_dete(exec_net, lm_exec_net, n, c, h, w, an, ac, ah, aw, label, frame, next_frame, \
                cur_request_id, next_request_id, loc):
    '''         6. process           '''
    attr_res = []
    (initial_h, initial_w, _) = frame.shape
    in_frame = cv2.resize(next_frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    exec_net.start_async(request_id=next_request_id, inputs={"data": in_frame})  #

    if exec_net.requests[cur_request_id].wait(-1) == 0:  # wait for done signal
        end1 = time.time()  # store the time
        # det_time = inf_end - inf_start
        res = exec_net.requests[cur_request_id].outputs["detection_out"]  # get the result
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > 0.6:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                if ((xmax<loc[0]) | (xmin>loc[2]) | (ymin>loc[3]) | (ymax<loc[1])):
                    cv2.rectangle(frame, (loc[0], loc[1]), (loc[2], loc[3]), (0, 255, 0), 10)
                else:
                    cv2.rectangle(frame, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 10)
                if xmin > 0 and ymin > 0 and (xmax < initial_w) and (ymax < initial_h):
                    roi = frame[ymin:ymax, xmin:xmax, :]
                    person_roi = cv2.resize(roi, (aw, ah))
                    person_roi = person_roi.transpose((2, 0, 1))
                    person_roi = person_roi.reshape((an, ac, ah, aw))
                    lm_exec_net.infer(inputs={"0": person_roi})  # start infer the person roi and wait for done
                    attr_res = lm_exec_net.requests[0].outputs["453"]
                    attrs = np.reshape(attr_res, (8, 1))
                    for i in range(len(attrs)):
                        if attrs[i][0] > 0.5:
                            cv2.putText(frame, str(label[i]) + ":1", (xmax, ymin + 20 * i), cv2.FONT_HERSHEY_COMPLEX,
                                        0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, str(label[i]) + ":0", (xmax, ymin + 20 * i), cv2.FONT_HERSHEY_COMPLEX,
                                        0.5, (0, 0, 255), 1)
                    cv2.putText(frame, "Person" + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    return frame


class DisplayWidget(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.args = build_argparser().parse_args()
        self.init_flag = 1
        self.source_flag = -1

        self.initUI()
        self.display()

    def initPara(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        person_model_xml = self.args.person_model
        person_model_bin = os.path.splitext(person_model_xml)[0] + ".bin"
        attribute_xml = self.args.attribute_model
        attribute_bin = os.path.splitext(attribute_xml)[0] + ".bin"
        face_dete_xml = self.args.face_dete_model
        face_dete_bin = os.path.splitext(face_dete_xml)[0] + ".bin"
        face_anyl_xml = self.args.face_anyl_model
        face_anyl_bin = os.path.splitext(face_anyl_xml)[0] + ".bin"
        fire_dete_xml = self.args.fire_model
        fire_dete_bin = os.path.splitext(fire_dete_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        self.plugin = IEPlugin(device=self.args.device, plugin_dirs=self.args.plugin_dir)
        if self.args.cpu_extension and 'CPU' in self.args.device:
            self.plugin.add_cpu_extension(self.args.cpu_extension)

        # Read IR
        log.info("Loading network:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}".format(self.args.person_model, self.args.attribute_model,self.args.face_dete_model,self.args.face_anyl_model,self.args.fire_model))
        self.person_dete_net = IENetwork(model=person_model_xml, weights=person_model_bin)
        self.person_attr_net = IENetwork(model=attribute_xml, weights=attribute_bin)
        self.face_dete_net = IENetwork(model=face_dete_xml, weights=face_dete_bin)
        self.face_anyl_net = IENetwork(model=face_anyl_xml, weights=face_anyl_bin)
        self.fire_dete_net = IENetwork(model=fire_dete_xml, weights=fire_dete_bin)

        '''          2.  set network affinity             '''
        if self.args.device == 'HETERO:FPGA,CPU':
            self.plugin.set_config({"TARGET_FALLBACK": self.args.device})  # set policy as fallback automatically
            self.plugin.set_initial_affinity(self.person_dete_net)
            self.plugin.set_initial_affinity(self.person_attr_net)
            self.plugin.set_initial_affinity(self.face_dete_net)
            self.plugin.set_initial_affinity(self.face_anyl_net)
            self.plugin.set_initial_affinity(self.fire_dete_net)
            self.firenet_affinity_setup()
            self.person_dete_affinity_setup()
            self.face_dete_affinity_setup()
            self.person_attr_affinity_setup()
            self.face_anyl_affinity_setup()

        # Read and pre-process input images
        if self.args.input == 'cam':
            self.input_stream = 0
        else:
            self.input_stream = self.args.input
            assert os.path.isfile(self.args.input), "Specified input file doesn't exist"

        self.cap = cv2.VideoCapture(WAIT_IMAGE)
        self.cur_request_id = 0
        self.next_request_id = 1
        _, self.frame = self.cap.read()
        self.labels = ["is_male", "has_bag", "has_backpack", "has_hat",
                       "has_longsleeves", "has_longpants", "has_longhair", "has_coat_jacket"]

    def firenet_affinity_setup(self):
        for l in self.fire_dete_net.layers.values():
            l.affinity = "FPGA"

    def person_attr_affinity_setup(self):
        for l in self.person_attr_net.layers.values():
            if(l.type != "Pooling") & (l.type != "sigmoid") & \
                (l.name != "0") & (l.name != "Mul_/Fused_Mul_/FusedScaleShift_") & \
                (l.name != "274") & (l.name != "275/mul_") & \
                (l.name != "276") & (l.name != "Mul1_1212/Fused_Mul_/FusedScaleShift_") & \
                (l.name != "278") & (l.name != "456") & (l.name != "454") :
                l.affinity = "FPGA"

    def face_anyl_affinity_setup(self):
        for l in self.face_anyl_net.layers.values():
            l.affinity = "FPGA"
    def person_dete_affinity_setup(self):
        for l in self.person_dete_net.layers.values():
            if (l.type == "Convolution") | (l.type == "ReLU"):
                l.affinity = "FPGA"
    def face_dete_affinity_setup(self):
        for l in self.face_dete_net.layers.values():
            if (l.type == "Convolution") | (l.type == "ReLU"):
                l.affinity = "FPGA"
    def initUI(self): 
        self.layout = QGridLayout()
        self.label = QLabel('显示区域')
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def display(self):
        '''用于显示的主函数'''
        self.main_window.loop_mode
        self.cap = cv2.VideoCapture("./wait.jpg")
        _, self.frame = self.cap.read()
        #TODO USER ID LIST
        self.user = []
        self.id = []
        # 创建一个workbook 设置编码
        self.workbook = wt.Workbook(encoding='utf-8')
        self.pass_alarm_area = (0,0,0,0)
        self.count = 0
        if os.path.exists(self.args.data):
            self.load_user_id()
        else:
            # 创建一个worksheet
            self.worksheet = self.workbook.add_sheet("My-Worksheet")
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(1000/30)
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self.check_loop)
        self.loop_timer.start(1000)

    def init_source(self,sig):
        '''
            TODO 请实现实际的视频源切换
                源0：摄像头实时拍摄内容
                源1：人脸检测
                源2：行人检测
                源3：火情检测
        '''
        if '开始' in sig:
            self.cap = cv2.VideoCapture(self.input_stream)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _, self.frame = self.cap.read()
        self.source_flag = 0
        self.init_flag = 1

    def regist(self, get_id):
        if self.source_flag == 1:
            self.user.append(self.attr_res)
            self.id.append(str(get_id))
            self.worksheet.write(0,self.count,str(get_id))
            face_time = time.strftime("-%y%m%d-%H_%M_%S", time.localtime())
            for i in range(512):
                self.worksheet.write(i+1,self.count,str(self.attr_res[0][i]))
            face_img = './face/'+str(get_id)+face_time
            cv2.imwrite(face_img + '.jpg', self.frame)
            self.workbook.save(self.args.data)
            self.count = self.count + 1
        else:
            print("当前不在人脸检测模式")

    def load_user_id(self):
        book = rd.open_workbook(self.args.data)
        sheet = book.sheets()[0]
        self.workbook = copy(wb=book)
        self.worksheet = self.workbook.get_sheet(0)
        nrows = sheet.nrows
        ncols = sheet.ncols
        print(nrows,ncols)
        self.count = ncols
        tmp = np.zeros([32,1,1,512],np.float32)
        for i in range(ncols):
            self.id.append(sheet.cell(0, i).value)
            for j in range(1,nrows):
                tmp[i][0][0][j-1] = float(sheet.cell(j,i).value)
            self.user.append(tmp[i])

    def alarm_dete(self, alarm_area):
        if self.source_flag == 2:
            xmin = int((alarm_area -1)%3 * 320)
            ymin = int((alarm_area -1)//3 * 360)
            self.pass_alarm_area = (xmin,ymin,xmin+320,ymin+360)
        else:
            print("当前不在行人检测模式")

    def check_loop(self):
        loop_mode = self.main_window.loop_mode
        if loop_mode:
            start_time = loop_mode[3]
            time_consumed = time.time() - start_time
            time_consumed_int = int(time_consumed)
            location_in_loop = time_consumed_int % (loop_mode[0] + loop_mode[1] + loop_mode[2])
            if location_in_loop <= loop_mode[0]:
                if self.source_flag != 1:
                    self.init_flag = 0
                self.source_flag = 1
            elif location_in_loop <= loop_mode[0] + loop_mode[1]:
                if self.source_flag != 2:
                    self.init_flag = 0
                self.source_flag = 2
            else:
                if self.source_flag != 3:
                    self.init_flag = 0
                self.source_flag = 3

    def refresh_display(self):
        if self.source_flag != -1:
           ret, next_frame = self.cap.read()
           if not ret:
              return

        start_time = time.time()

        if self.source_flag == 0:
            self.label.setPixmap(QPixmap())
        elif self.source_flag == 1:
            if self.init_flag == 0:
                self.exec_net, self.lm_exec_net, self.n, self.c, self.h, \
                self.w, self.an, self.ac, self.ah, self.aw = double_init(self.plugin, self.face_dete_net,
                                                                         self.face_anyl_net)
                self.init_flag = 1
            self.frame, self.attr_res = face_dete(self.exec_net, self.lm_exec_net, self.n, self.c, self.h, self.w, self.an,
                                             self.ac, self.ah, self.aw,
                                             self.frame, next_frame, self.cur_request_id, self.next_request_id,
                                             self.user, self.id)
        elif self.source_flag == 2:
            if self.init_flag == 0:
                self.exec_net, self.lm_exec_net, self.n, self.c, self.h, \
                self.w, self.an, self.ac, self.ah, self.aw = double_init(self.plugin, self.person_dete_net,
                                                                         self.person_attr_net)
                self.init_flag = 1
            self.frame = person_dete(self.exec_net, self.lm_exec_net, self.n, self.c, self.h, self.w, self.an, self.ac,
                                     self.ah, self.aw, self.labels, self.frame, next_frame, self.cur_request_id, \
                                     self.next_request_id,self.pass_alarm_area)
        elif self.source_flag == 3:
            if self.init_flag == 0:
                self.exec_net, self.n, self.c, self.h, self.w = single_init(self.plugin, self.fire_dete_net)
                self.init_flag = 1
            self.frame = fire_dete(self.exec_net, self.n, self.c, self.h, self.w,
                                   self.frame, next_frame, self.cur_request_id, self.next_request_id)
        infer_time = time.time() - start_time
        infer_time_message = "Inference time: {:.3f} ms".format(infer_time*1e3)
        cv2.putText(self.frame,infer_time_message, (15,15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        show = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.source_flag != -1:
           self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
           self.frame = next_frame

        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))
if __name__ == '__main__':
    pass

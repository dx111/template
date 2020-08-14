import re
import cv2
import json
import itertools
import numpy as np
from hungarian import Hungarian
from xml.etree import ElementTree
from typing import List, Dict

anchor_json_file = "template/sp/anchor.json"
key_json_file = "template/sp/key.json"
position_xml_file = "template/sp/1.xml"
template_jpg_file = "C:\\Users\\dx\\PycharmProjects\\template_ocr\\template\\sp\\template.jpg"
Debug = False
from PIL import Image
img=Image.Image.


class SpecialInvoice(object):
    def __init__(self, anchor_file: str, key_file: str, position_file: str):
        self.anchor = self.load_anchor(anchor_file, position_file)
        self.key = self.load_key(key_file, position_file)

    @staticmethod
    def cal_iou(box1: List[int], box2: List[int]):
        """
        Calculate the IOU of two boxes

        :return:
        :param box1: [x_min1, y_min1, x_max1, y_max1]
        :param box2: [x_min2, y_min2, x_max2, y_max2]
        :return: float
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        s1 = (x1_max - x1_min) * (y1_max - y1_min)
        s2 = (x2_max - x2_min) * (y2_max - y2_min)

        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        w = max(0, x_max - x_min)
        h = max(0, y_max - y_min)
        area = w * h
        iou = area / (s1 + s2 - area)
        return iou

    def load_anchor(self, anchor_file: str, position_file: str) -> dict:
        """
        load anchor file

        :param anchor_file: str
        :param position_file: str
        :return: anchor
        """
        position = self.load_position(position_file)
        with open(anchor_file, "r", encoding="utf-8") as f:
            anchor = json.load(f)
        for key in anchor.keys():
            anchor[key]["position"] = position[key]
        return anchor

    def load_key(self, key_file: str, position_file: str) -> dict:
        """
        load key info

        :param key_file: str
        :param position_file: str
        :return: key info
        """
        position = self.load_position(position_file)
        with open(key_file, "r", encoding="utf-8") as f:
            key_info = json.load(f)
        for k in key_info.keys():
            key_info[k]["position"] = position[k]
        return key_info

    @staticmethod
    def load_position(position_file: str) -> dict:
        """
        load xml file

        :param position_file: str
        :return: dict
        """
        et = ElementTree.parse(position_file)
        objs = et.findall("object")
        position = {}
        for obj in objs:
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            position[name] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        return position

    def get_perspective_box(self, text_list: list, poly_list: list):
        src_boxes = []
        dst_boxes = []
        for a_key in self.anchor.keys():
            match_list = map(lambda text: re.match(self.anchor[a_key]["regex"], text), text_list)
            match_index_list = [ind for ind, match in enumerate(match_list) if match]
            if len(match_index_list) == 1:
                polys = np.array(poly_list[match_index_list[0]], dtype=np.float)
                src_boxes.append(polys)
                dst_boxes.append(self.anchor[a_key]["position"])
        return src_boxes, dst_boxes

    @staticmethod
    def get_best_perspective_transform(src_boxes: list, dst_boxes: list) -> np.ndarray:
        min_distance = 1e7
        best_matrix = None
        all_index_list = range(len(src_boxes))
        combination = itertools.combinations(all_index_list, 4)
        for c in combination:
            src_box_p = np.array([src_boxes[y][0] for y in c], dtype=np.float32)
            dst_box_p = np.array([dst_boxes[y][0] for y in c], dtype=np.float32)

            c_extra = set(all_index_list).difference(set(c))
            src_box_extra = np.array([src_boxes[y] for y in c_extra], np.float)
            dst_box_extra = np.array([dst_boxes[y] for y in c_extra], np.float)

            transform_matrix = cv2.getPerspectiveTransform(src_box_p, dst_box_p)
            src_box_extra = cv2.perspectiveTransform(src_box_extra, transform_matrix)
            distance = np.linalg.norm(np.array(src_box_extra).reshape(-1, 2) - dst_box_extra.reshape(-1, 2), ord=2)
            if Debug:
                result = cv2.imread(template_jpg_file)
                src_boxes_show = cv2.perspectiveTransform(np.array(src_boxes, np.float), transform_matrix)
                cv2.polylines(result, np.array(src_boxes_show).astype("int"), 1, (0, 255, 0), 3)
                cv2.imwrite("debug\\m\\%f.jpg" % distance, result)
            if distance < min_distance:
                min_distance = distance
                best_matrix = transform_matrix

        return best_matrix

    def get_cost_matrix(self):
        pass

    def structure(self, ocr_result) -> dict:
        text_list = [res["text"] for res in ocr_result]
        poly_list = [res["quadrangle"] for res in ocr_result]
        src_boxes, dst_boxes = self.get_perspective_box(text_list, poly_list)
        transform_matrix = self.get_best_perspective_transform(src_boxes, dst_boxes)
        dst = cv2.perspectiveTransform(np.array(poly_list), transform_matrix)
        if Debug:
            img = cv2.imread(template_jpg_file)
            cv2.drawContours(img, dst.astype(np.int), -1, (0, 255, 0), 1)
            cv2.imwrite("debug/t/output.jpg", img)

        cost_matrix = np.zeros(shape=(np.array(poly_list).shape[0], len(self.key.keys())))
        key_index = {}
        for i, d in enumerate(dst):
            x1_min, y1_min, y1_max = d[0][0], d[0][1], d[2][1]
            pts_1 = np.array([[x1_min, y1_min], [x1_min, y1_max]])
            # pts_1 = np.array(d, np.float).reshape((-1, 2))
            for j, ak in enumerate(self.key.keys()):
                key_index[j] = self.key[ak]["name"]
                anchor_position = self.key[ak]["position"]
                x2_min, y2_min, y2_max = anchor_position[0][0], anchor_position[0][1], anchor_position[2][1]
                pts_2 = np.array([[x2_min, y2_min], [x2_min, y2_max]])
                # pts_2 = np.array(self.key[ak]["position"], np.float).reshape((-1, 2))
                distance = np.linalg.norm(pts_1 - pts_2, ord=2)
                cost_matrix[i, j] = distance

        # print(cost_matrix)
        # np.savetxt("cost.txt", cost_matrix, fmt="%d", delimiter=" ")

        hungarian = Hungarian()
        hungarian.calculate(cost_matrix)
        res = hungarian.get_results()
        struct_dict = dict()
        print(res)
        for _p in res:
            struct_dict[key_index[_p[1]]] = text_list[_p[0]]
            print(key_index[_p[1]], text_list[_p[0]], cost_matrix[_p[0], _p[1]])
        struct_dict = self.post_process(struct_dict)
        return struct_dict

    @staticmethod
    def post_process(struct_dict: dict) -> dict:
        for key in struct_dict.keys():
            if key == "购买方名称" or key == "销售方名称":
                struct_dict[key] = struct_dict[key].replace("称：", "")
            if key == "税额" or key == "价税合计" or key == "税前总额":
                struct_dict[key] = struct_dict[key].replace("￥", "")
            if key == "购买方地址" or key == "销售方地址":
                struct_dict[key] = struct_dict[key].replace("电话：", "")
            if key == "购买方开户行" or key == "销售方开户行":
                struct_dict[key] = struct_dict[key].replace("开户行及账号：", "")
            if key == "价税合计":
                struct_dict[key] = struct_dict[key].replace("（", "")
                struct_dict[key] = struct_dict[key].replace("小", "")
                struct_dict[key] = struct_dict[key].replace("写", "")
                struct_dict[key] = struct_dict[key].replace("）", "")
                struct_dict[key] = struct_dict[key].replace("?", "")

        return struct_dict


sp = SpecialInvoice(anchor_json_file, key_json_file, position_xml_file)

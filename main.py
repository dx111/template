import requests
import base64
import time
from invocie_template import sp

proxies = {
    'http': 'http://10.110.63.24:7890',
    'https': 'https://10.110.63.24:7890'
}


def request_ocr(image_file):
    with open(image_file, "rb") as f:
        base64_data = str(base64.b64encode(f.read()), encoding="utf8")
        r = requests.post(url="http://10.110.63.24:8088/ocr", json={'base64_strs': base64_data})
        result = r.json()["res"]
        return result


if __name__ == "__main__":
    img_file = "C:\\Users\\dx\\PycharmProjects\\template_ocr\\test_images\\5.jpg"
    ocr_result = request_ocr(img_file)
    time_start = time.time()
    struct = sp.structure(ocr_result)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print(struct)

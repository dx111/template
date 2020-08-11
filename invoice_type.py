import re

general_str = "普通普通"
electronic_str = "电子发票"
special_str = "专用发票"


def get_invoice_type(ocr_result):
    flag_electronic = False
    flag_general = False
    flag_special = False
    for result in ocr_result:
        text = result["text"]
        flag_general = True if re.match(".*普通发票$", text) else flag_general
        flag_electronic = True if re.match("(?=.*电子)^.*发票$", text) else flag_electronic
        flag_special = True if re.match(".*专用发票$", text) else flag_special

    if flag_electronic:
        return electronic_str
    elif flag_special:
        return special_str
    elif flag_general:
        return general_str
    else:
        "未识别到发票类型"

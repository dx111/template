from pydantic import BaseSettings


class Config(BaseSettings):
    Debug = True
    img_file = "C:\\Users\\dx\\PycharmProjects\\template_ocr\\test_images\\4.jpg"
    tmp_file = "C:\\Users\\dx\\PycharmProjects\\template_ocr\\template\\sp\\template.jpg"


cfg = Config()

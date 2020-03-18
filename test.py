# -*- coding:utf-8 -*-
#######APIµ÷ÓÃ##########
import requests
input_img_pth = 'your img path'  ##example:  './img/1.jpg'

upload_url = 'https://momodel.cn/pyapi/file/temp_api_file'
img_file = {'file': open(input_img_pth, 'rb')}
img_file_name = requests.post(upload_url, files=img_file).json().get('temp_file_name')
    
base_url = "https://momodel.cn/pyapi/apps/run/"
app_id = "5e267a93d13fba905e3323be"
input_dic = {"img": {"val": img_file_name, "type": "img"}}
output_dic = {"str": {"type": "str"}}
app_version = "dev"
payload = {"app": {"input": input_dic, "output": output_dic}, "version": app_version}
response = requests.post(base_url + app_id, json=payload)
#print(response.json())  ##the img bs64 str results

########base64 to img ########
import re
import base64
from io import BytesIO
import numpy as np
from PIL import Image

save_pth = 'img path to save' ##example:  './results/1.jpg'

def base64_to_image(base64_str,save_path):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save(save_path)
    return img

result_img=base64_to_image(response.json()['response']['str'],save_pth)
result_img.show()
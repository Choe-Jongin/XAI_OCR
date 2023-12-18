from ocr import *

with open("label.data", "r") as json_file:
    json_data = json.load(json_file)
    
for box in json_data['boundary']:
    print("["+box['text'].replace("\n",""), box['block_num'],box['left_top'], box['right_bot'], sep="][", end="]\n")
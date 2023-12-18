from ocr import *

dirname = "images/model/"
files = os.listdir(dirname)
if dirname[-1] != "/":
    dirname = dirname+"/"
files = [dirname + file for file in files]

index = 0
for file in files:
    if ".pdf" in file or "PDF" in file:
        image_file_name = file.replace(".pdf", ".png").replace(".PDF", ".png")
        # convert to jpg
        if not os.path.isfile(image_file_name):
            print("convert pdf to png")
            pages = convert_from_path(file, poppler_path=r'C:\poppler\Library\bin')
            print("save", image_file_name)
            pages[0].save(dirname+"model"+str(index)+".png", "png")
        file = image_file_name
    print("read file :", file)
    index += 1
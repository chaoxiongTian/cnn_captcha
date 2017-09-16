import os
import shutil

IMAGE_MUMBER = 300
IMAGE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/baidu_image_captcha/"
LABEL_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/"
LABEL_FILE_NAME = "baidu_labels_captcha.txt"
COPE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/dev/"
if not os.path.exists(COPE_PATH):
    os.makedirs(COPE_PATH)
IMAGE_FORMAT = ".jpg"


def getStrContent(path):
    return open(path, 'r').read()


def save_image(image_list):
    for each in range(len(image_list)):
        path1 = image_list[each]
        path2 = COPE_PATH + str(each) + IMAGE_FORMAT
        cope_image(path1, path2)


def cope_image(file_path_original, file_path_target):
    shutil.copy(file_path_original, file_path_target)


def save_labels(labels_list):
    content = '#'.join(labels_list)
    print(content)
    f = open(COPE_PATH + "dev_labels_text.txt", 'w')
    f.write(content)
    f.close()


def cope_file(image_list, labels_list):
    save_image(image_list)
    save_labels(labels_list)


def main():
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    label_path = label_path + LABEL_FILE_NAME
    images_path = []
    for each in range(IMAGE_MUMBER):
        images_path.append(image_path + str(each) + IMAGE_FORMAT)
    string = getStrContent(label_path)
    labels = string.split("#")
    cope_file(images_path[-IMAGE_MUMBER:], labels[-IMAGE_MUMBER:])


if __name__ == '__main__':
    main()

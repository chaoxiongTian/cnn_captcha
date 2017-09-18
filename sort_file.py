import os
import shutil

IMAGE_MUMBER_ALL = 34 #一共有多少个图片
IMAGE_MUMBER_CPOE = 5 #需要拷贝多少图片
IMAGE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/error_image/"
LABEL_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/"
LABEL_FILE_NAME = "error_labels_text.txt"
COPE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/dev/"
if not os.path.exists(COPE_PATH):
    os.makedirs(COPE_PATH)
IMAGE_FORMAT = ".jpg"


def getStrContent(path):
    return open(path, 'r').read().strip()


def save_image(image_list):
    for each in range(len(image_list)):
        path1 = image_list[each]+ IMAGE_FORMAT
        path2 = COPE_PATH + str(each) + IMAGE_FORMAT
        print(path1)
        print(path2)
        cope_image(path1, path2)


def cope_image(file_path_original, file_path_target):
    # shutil.cope(file_path_original, file_path_target)
    shutil.move(file_path_original, file_path_target)


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
    for each in range(IMAGE_MUMBER_ALL):
        images_path.append(image_path + str(each))
    string = getStrContent(label_path)
    labels = string.split("#")
    cope_file(images_path[-IMAGE_MUMBER_CPOE:], labels[-IMAGE_MUMBER_CPOE:])


if __name__ == '__main__':
    main()

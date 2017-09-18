import os
import shutil

IMAGE_MUMBER_ALL = 5  # 一共有多少个图片
IMAGE_MUMBER_MOVE = 4  # 需要拷贝多少图片
IMAGE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/ceshi/"
LABEL_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/"
LABEL_FILE_NAME = "ceshi.txt"
COPE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/mv_image/"


# 清空文件夹原来的内容
def removeFileInFirstDir(target_dir):
    for file in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file)
        if os.path.isfile(target_file):
            os.remove(target_file)


if not os.path.exists(COPE_PATH):
    os.makedirs(COPE_PATH)
else:
    removeFileInFirstDir(COPE_PATH)
IMAGE_FORMAT = ".jpg"


def getStrContent(path):
    return open(path, 'r').read().strip().split("#")


def save_image(image_list):
    for each in range(len(image_list)):
        cope_image(image_list[each], COPE_PATH + str(each) + IMAGE_FORMAT)


def cope_image(file_path_original, file_path_target):
    shutil.move(file_path_original, file_path_target)


def save_labels(labels_list, save_path):
    f = open(save_path, 'w')
    f.write('#'.join(labels_list))
    f.close()


def cope_file(image_list, labels_list):
    save_image(image_list)
    save_labels(labels_list, COPE_PATH + "mv_image_labels.txt")


def main():
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    label_path = label_path + LABEL_FILE_NAME

    images_path = [image_path + str(x) + IMAGE_FORMAT for x in range(IMAGE_MUMBER_ALL)]
    labels = getStrContent(label_path)
    cope_file(images_path[-IMAGE_MUMBER_MOVE:], labels[-IMAGE_MUMBER_MOVE:])
    # 剪切labels 如果复制不需要执行这个
    save_labels(labels[:-IMAGE_MUMBER_MOVE], label_path)


if __name__ == '__main__':
    main()

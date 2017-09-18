import os
import shutil


"""
从制定文件夹中拷贝最后的多少个图片到制定文件中
"""
IMAGE_MUMBER_ALL = 2  # 一共有多少个图片
IMAGE_MUMBER_MOVE = 1  # 需要拷贝多少图片
IMAGE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/ceshi/"
LABEL_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/"
LABEL_FILE_NAME = "ceshi.txt"
COPE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/mv_image/"
try:
    IMAGE_FORMAT = os.listdir(IMAGE_PATH)[1][-4:]
except IOError as err:
        print("something error:\n" + str(err))
        print("确定后缀格式，文件是不是只有图片格式文件")


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

    # 这里不通过os.listdir去读写文件是因为需要和labels对应起来。
    images_path = [image_path + str(x) + IMAGE_FORMAT for x in range(IMAGE_MUMBER_ALL)]
    labels = getStrContent(label_path)
    cope_file(images_path[-IMAGE_MUMBER_MOVE:], labels[-IMAGE_MUMBER_MOVE:])
    # 剪切labels 如果复制不需要执行这个
    save_labels(labels[:-IMAGE_MUMBER_MOVE], label_path)


if __name__ == '__main__':
    main()

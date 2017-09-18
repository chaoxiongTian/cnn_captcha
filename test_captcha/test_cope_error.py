import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image

# 图像大小
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63
IMAGE_FORMAT = ".jpg"

IMAGE_MUMBER = 10
IMAGE_PATH = "../datasets/test_sets/"
LABEL_PATH = "../datasets/"
LABEL_FILE_NAME = "test_labels.txt"

DIR_ERROR_IMAGES = "../datasets/error_image/"
if not os.path.exists(DIR_ERROR_IMAGES):
    os.makedirs(DIR_ERROR_IMAGES)

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([32 * 32 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def path_np_image(image_path):
    captcha_image = Image.open(image_path)
    captcha_image = np.array(captcha_image)
    captcha_image = convert2gray(captcha_image)
    captcha_image = captcha_image.flatten() / 255
    return captcha_image


def crack_captcha_images(images, lables):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    error_images_index = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        num = 0
        error_lables = []
        error_images = []
        for each in range(len(images)):

            text_list = sess.run(predict, feed_dict={X: [images[each]], keep_prob: 1})

            text = text_list[0].tolist()  # [2, 8, 49, 50]
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1
            print("正确: {}  预测: {}".format(lables[each], vec2text(vector)))
            # if lables[each].lower() == vec2text(vector).lower(): #　忽略大小写
            if lables[each] == vec2text(vector):
                num = num + 1
            else:
                error_images_index.append(each)
                error_lables.append(lables[each])
                error_images.append(vec2text(vector))
        return num / len(images), error_lables, error_images, error_images_index


def getStrContent(path):
    return open(path, 'r').read().strip()





def compare_two_str(str1, str2, error_lables_char_list, error_images_char_list, error_images_char_number_list):
    for each in range(len(str1)):
        flag = 0

        char1 = str1[each]
        char2 = str2[each]
        if char1 != char2:
            for each2 in range(len(error_lables_char_list)):
                char3 = error_lables_char_list[each2]
                char4 = error_images_char_list[each2]
                if char1 == char3:
                    if char2 == char4:
                        flag = 1
                        error_images_char_number_list[each2] = error_images_char_number_list[each2] + 1

            if flag == 0:
                error_images_char_number_list.append(1)
                error_lables_char_list.append(str1[each])
                error_images_char_list.append(str2[each])


def compare_error_log(error_lables, error_images, error_lables_char_list, error_images_char_list,
                      error_images_char_number_list):
    for each in range(len(error_lables)):
        if error_lables[each] == error_images[each]:
            pass
        else:
            compare_two_str(error_lables[each], error_images[each]
                            , error_lables_char_list, error_images_char_list, error_images_char_number_list)


def sort(list1, list2, list3):
    for i in range(len(list1) - 1):
        for j in range(len(list1) - i - 1):  # ｊ为列表下标
            if list1[j] > list1[j + 1]:
                list1[j], list1[j + 1] = list1[j + 1], list1[j]
                list2[j], list2[j + 1] = list2[j + 1], list2[j]
                list3[j], list3[j + 1] = list3[j + 1], list3[j]


def sort2(list1, list2, list3):
    for i in range(len(list1) - 1):
        for j in range(len(list1) - i - 1):  # ｊ为列表下标
            if list3[j] < list3[j + 1]:
                list1[j], list1[j + 1] = list1[j + 1], list1[j]
                list2[j], list2[j + 1] = list2[j + 1], list2[j]
                list3[j], list3[j + 1] = list3[j + 1], list3[j]


def cope_image(file_path_original, file_path_target):
    shutil.copy(file_path_original, file_path_target)


def cope_error_image_2_other_dir(error_images_index, labels):
    dir_error_images = DIR_ERROR_IMAGES
    dir_test_image = IMAGE_PATH
    error_labels = []
    for each in range(len(error_images_index)):
        error_labels.append(labels[error_images_index[each]])
        # print(labels[error_images_index[each]])
        file_path_original = dir_test_image + str(error_images_index[each]) + IMAGE_FORMAT

        file_path_target = dir_error_images + str(each) + IMAGE_FORMAT

        cope_image(file_path_original, file_path_target)
        error_labels_text = '#'.join(error_labels)

        f = open(LABEL_PATH + "error_labels_text.txt", 'w')
        f.write(error_labels_text)
        f.close()


def main():
    image_path = IMAGE_PATH  # 用来测试图片的存储路径
    label_path = LABEL_PATH  # 用来测试labels的存储路径
    label_path = label_path + LABEL_FILE_NAME
    images_path = []
    images = []
    for each in range(IMAGE_MUMBER):
        images_path.append(image_path + str(each) + IMAGE_FORMAT)

    string = getStrContent(label_path)
    labels = string.split("#")
    # images_path : 图片路劲的 list 更加存储路径才可取出操作
    # labels : label 用来对比算准确率
    # images:根据images_path拿出的图像之后转np格式,转灰度/255之后的list
    for each in images_path:
        image = path_np_image(each)
        images.append(image)

    # accuracy : 错误率
    # error_labels : 错误的labels list
    # error_images : 错误的image list
    # error_images_index : 错误的图片的下标
    accuracy, error_labels, error_images, error_images_index = crack_captcha_images(images, labels)

    print("错误预测有:")
    for each in range(len(error_labels)):
        print("正确: {}  预测: {}".format(error_labels[each], error_images[each]))
    """
    构建一种把谁预测成谁的输出
    应该出现的  识别错的  识别错的概率
    b 9 2
    3 Z 1
    H s 1
    """
    error_labels_char_list = []
    error_images_char_list = []
    error_images_char_number_list = []

    compare_error_log(error_labels, error_images, error_labels_char_list, error_images_char_list,
                      error_images_char_number_list)
    # 两个排序
    # 先按照字符的ASCII排序 然后按照出错的频率排序
    sort(error_labels_char_list, error_images_char_list, error_images_char_number_list)
    sort2(error_labels_char_list, error_images_char_list, error_images_char_number_list)

    for i in range(len(error_labels_char_list)):
        print(error_labels_char_list[i] +" " +
              error_images_char_list[i] + " " +
              str(error_images_char_number_list[i]))
    print("准确率:" + str(1 - len(error_labels) / IMAGE_MUMBER))
    print("错误个数:" + str(len(error_labels)))
    # 把识别错误的图片重新整理.
    print(error_images_index)
    cope_error_image_2_other_dir(error_images_index, labels)


if __name__ == '__main__':
    main()

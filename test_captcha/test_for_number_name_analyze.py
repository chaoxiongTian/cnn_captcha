import numpy as np
import tensorflow as tf
from PIL import Image

# 图像大小
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63


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

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

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
            if lables[each].lower() == vec2text(vector).lower():
                num += 1
            else:
                error_lables.append(lables[each])
                error_images.append(vec2text(vector))
        return num / len(images), error_lables, error_images


def getStrContent(path):
    return open(path, 'r').read()


IMAGE_NUMBER = 20
IMAGE_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/four_char_clean/B/test/"
LABEL_PATH = "/home/tianchaoxiong/LinuxData/data/verifies/full_experiment/four_char_clean/B/"
LABEL_FILE_NAME = "text_four_char_for_full_experiment_20.txt"


def compare_two_str(str1, str2, error_labels_char_list, error_images_char_list, error_images_char_number_list):
    for each in range(len(str1)):
        flag = 0
        char1 = str1[each]
        char2 = str2[each]
        if char1.lower() != char2.lower():
            for each2 in range(len(error_labels_char_list)):
                char3 = error_labels_char_list[each2]
                char4 = error_images_char_list[each2]
                if char1 == char3:
                    if char2 == char4:
                        flag = 1
                        error_images_char_number_list[each2] += 1

            if flag == 0:
                error_images_char_number_list.append(1)
                error_labels_char_list.append(str1[each])
                error_images_char_list.append(str2[each])


def compare_error_log(error_labels, error_images, error_labels_char_list, error_images_char_list,
                      error_images_char_number_list):
    for each in range(len(error_labels)):
        if error_labels[each] == error_images[each]:
            pass
        else:
            compare_two_str(error_labels[each], error_images[each]
                            , error_labels_char_list, error_images_char_list, error_images_char_number_list)


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


def get_threshold_index(error_images_char_number_list, correct_number_threshold):
    index = -1
    for each in range(len(error_images_char_number_list)):
        if int(error_images_char_number_list[each]) > correct_number_threshold:
            index += 1
    return index


def search_correct_char(char_predict, error_labels_char_list, error_images_char_list, error_number_list):
    # print(char_predict)
    for each in range(len(error_images_char_list)):
        if char_predict == error_images_char_list[each]:
            return error_labels_char_list[each], error_number_list[each]
    return "_", -1


def correct_error(error_labels, error_images,
                  error_labels_char_list, error_images_char_list, error_number_list):
    correct_number = 0
    for each in range(len(error_labels)):
        label = error_labels[each]
        predict = error_images[each]
        correct_chars = []
        correct_priority = []
        correct_index = []
        for each2 in range(len(label)):
            char_label = label[each2]
            char_predict = predict[each2]
            if char_label.lower() == char_predict.lower():
                pass
            else:
                correct_char, priority = search_correct_char(char_predict, error_labels_char_list,
                                                             error_images_char_list, error_number_list)
                correct_chars.append(correct_char)
                correct_priority.append(priority)
                correct_index.append(each2)
        for each3 in range(len(correct_chars)):
            if correct_chars[each3] != '_':
                print("predict: %s" % predict)
                print("correct_index: %s" % correct_index[each3])
                print("correct_char: %s" % correct_chars[each3])
                correct_predict = replace_error_char(predict, correct_chars, correct_index, each3)
                # predict[correct_index[each3]] = correct_chars[correct_index[each3]]
                print("correct_predict: %s\t label: %s" % (correct_predict, label))
                if label.lower() == correct_predict.lower():
                    correct_number += 1
                    break

    return correct_number


def replace_error_char(predict, correct_chars, correct_index, each3):
    if correct_index[each3] == 0:
        correct_predict = correct_chars[each3] + predict[(correct_index[each3] + 1):]
    elif correct_index[each3] == len(predict) - 1:
        correct_predict = predict[:correct_index[each3]] + correct_chars[each3]
    else:
        correct_predict = predict[:correct_index[each3]] + correct_chars[each3] \
                          + predict[(correct_index[each3] + 1):]
    return correct_predict


def main():
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    label_path += LABEL_FILE_NAME
    images_path = []
    images = []
    error_labels = []
    error_images = []
    error_labels_images = []
    labels = []
    file_predict_errors = open('./predict_errors.txt', 'w+')
    for each in range(IMAGE_NUMBER):
        images_path.append(image_path + str(each + 10000 - IMAGE_NUMBER) + ".png")
    string = getStrContent(label_path)
    labels_all = string.split("#")

    for each in images_path:
        image = path_np_image(each)
        images.append(image)
    for each in range(IMAGE_NUMBER):
        label = labels_all[each + 10000 - IMAGE_NUMBER]
        labels.append(label)
    accuracy, error_labels, error_images = crack_captcha_images(images, labels)
    print("错误预测有:")
    for each in range(len(error_labels)):
        print("正确: {}  预测: {}".format(error_labels[each], error_images[each]))
        error_str = error_labels[each] + "\t" + error_images[each] + "\n"
        error_labels_images.append(error_str)
    print("预测错误验证码写入文件：\n")
    file_predict_errors.writelines(error_labels_images)
    print("准确率:" + str(1 - len(error_labels) / IMAGE_NUMBER))

    error_labels_char_list = []
    error_images_char_list = []
    error_images_char_number_list = []
    error_char_statics = []
    compare_error_log(error_labels, error_images, error_labels_char_list, error_images_char_list,
                      error_images_char_number_list)
    sort(error_labels_char_list, error_images_char_list, error_images_char_number_list)

    sort2(error_labels_char_list, error_images_char_list, error_images_char_number_list)
    for i in range(len(error_labels_char_list)):
        char_statics = error_labels_char_list[i] + " " \
                       + error_images_char_list[i] + " " + str(error_images_char_number_list[i]) + "\n"
        print(char_statics)
        error_char_statics.append(char_statics)
    print("将识别错误字符写入文件：\n")
    file_predict_errors.writelines(error_char_statics)
    file_predict_errors.close()

    correct_number_threshold = 5
    threshold_index = get_threshold_index(error_images_char_number_list, correct_number_threshold)
    correct_number = correct_error(error_labels, error_images, error_labels_char_list[0:threshold_index],
                                   error_images_char_list[0:threshold_index],
                                   error_images_char_number_list[0:threshold_index])
    print("correct_number=%d" % correct_number)

    print("准确率:" + str(1 - len(error_labels) / IMAGE_NUMBER))
    print("准确率:" + str(1 - (len(error_labels) - correct_number) / IMAGE_NUMBER))


if __name__ == '__main__':
    main()

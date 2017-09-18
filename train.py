# -- coding: UTF-8 --
import numpy as np
import tensorflow as tf
import random
import os
from PIL import Image

# 图像大小
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_CAPTCHA = 4
CHAR_SET_LEN = 63

# 文件夹中文件的个数
IMAGE_MUMBER = 20
EPOCH = 200
BATCH_SIZE = 10
IMAGE_PATH = "datasets/train_sets/"
LABEL_PATH = "datasets/"
LABEL_FILE_NAME = "train_labels.txt"
MODEL_SAVE_PATH = "checkpoints/models/"

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


if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
else:
    removeFileInFirstDir(MODEL_SAVE_PATH)


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


# 生成一个训练batch
def get_next_batch(batch_size, each, images, labels):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def get_captcha_text_and_image(i, each):
        image_num = each * batch_size + i
        label = labels[image_num]
        image_path = images[image_num]
        # print(str(each)+" "+str(image_num)+" "+label+" "+image_path)
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image

    for i in range(batch_size):
        text, image = get_captcha_text_and_image(i, each)

        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


# 随机生成一个训练batch
def get_random_batch(batch_size, images, labels):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def get_captcha_text_and_image(i):
        image_num = i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image

    for i in range(batch_size):
        text, image = get_captcha_text_and_image(random.randint(0, IMAGE_MUMBER - 1))

        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


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


# 训练
# images, labels 图片路径和对应标签的list
def train_crack_captcha_cnn(images, labels):
    # 定义网络
    output = crack_captcha_cnn()

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCH):
            # 每个epoch
            for each in range(int(IMAGE_MUMBER / BATCH_SIZE)):
                # print(str(int(IMAGE_MUMBER / BATCH_SIZE)))
                # print("第几次:"+str(each))
                batch_x, batch_y = get_next_batch(BATCH_SIZE, each, images, labels)
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                # print(epoch, loss_)
                print("epoch: %d  iter: %d/%d   loss: %f" % (epoch + 1, BATCH_SIZE * each, IMAGE_MUMBER, loss_))

            # # 计算准确率 可用训练集 也可用测试集合
            batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, images, labels)
            acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
            # print(epoch, acc)
            print("epoch: %d  acc: %f" % (epoch + 1, acc))
            if (epoch + 1) % 5 == 0:
                saver.save(sess, MODEL_SAVE_PATH + "crack_capcha.model", global_step=epoch + 1)


def getStrContent(path):
    return open(path, 'r', encoding="utf-8").read().strip().split("#")


def main():
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    label_path = label_path + LABEL_FILE_NAME

    # 生成image_path list
    images = [image_path + str(x) + IMAGE_FORMAT for x in range(IMAGE_MUMBER)]
    # 生成labels list
    labels = getStrContent(label_path)
    train_crack_captcha_cnn(images, labels)


if __name__ == '__main__':
    main()

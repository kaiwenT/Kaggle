# Logistic Regression based on Kaggle Tinanic data
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


# 从csv文件读取数据
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = \
        tf.train.string_input_producer([file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

# 参数初始化
w = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# 值合并，也就是计算推断函数
def combine_inputs(X):
    return tf.matmul(X, w) + b


# 计算模型在X上的输出
def inference(X):
    return tf.sigmoid(combine_inputs(X))


loss_set = []
b_set = []


# 计算损失
def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


# 读取或生成训练数据X，及期望输出Y
def inputs():
    record_defaults = [[0.], [0.], [0.], [""], [""], [0.], [0.], [0.], [""], [0.], [""], [""]]

    passenger_id, survived, pclass, name, sex, age, \
    sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(891, "E:/机器学习/泰坦尼克/train.csv", record_defaults)

    # 转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [891, 1])

    return features, survived


# 依据计算的总损失 训练、调整模型参数
def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


# 对训练得到的模型进行评估
def evaluate(sess, X, Y):
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print('准确率：', sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# 创建一个Saver对象
saver = tf.train.Saver()

# 在一个会话中启动数据流图
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 训练迭代次数
    training_steps = 1000
    try:
        initial_step = 0
        # 验证之前是否保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__) + '/checkpoints/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

        # 训练闭环
        for step in range(initial_step, training_steps):
            if coord.should_stop():
                break
            sess.run([train_op])
            # 查看损失函数的值
            if step % 10 == 0:
                print("loss: ", sess.run(total_loss), "b: ", sess.run(b))
                loss_set.append(sess.run(total_loss))
                b_set.append(sess.run(b))

            # if step % 1000 == 0:
            #     saver.save(sess, 'logregression-model', global_step=step)

    except Exception as e:
        coord.request_stop(e)

    evaluate(sess, X, Y)

    # saver.save(sess, 'logregression-model', global_step=training_steps)

    coord.request_stop()
    coord.join(threads)
    sess.close()

# 画图显示b和loss的关系
plt.plot(b_set, loss_set, 'o-', ms=3, lw=1.5, color='green')
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$loss$', fontsize=16)
plt.show()

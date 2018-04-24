import tensorflow as tf


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

#计算模型在X上的输出
def inference(X):

# 计算损失
def loss(X, Y):

# 读取或生成训练数据X，及期望输出Y
def inputs():

# 依据计算的总损失训练、调整模型参数
def train(total_loss):

# 对训练得到的模型进行评估
def evaluate(sess, X, Y):

# 在一个会话中启动数据流图
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        #
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()


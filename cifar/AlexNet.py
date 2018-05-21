import pickle
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


# 解压数据文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic[b'data'], dic[b'labels']


def train_test_split(file):
    train_X = []
    train_y = []

    for i in range(1, 5):
        images, labels = unpickle(file + str(i))
        train_X.extend(images)
        train_y.extend(labels)

    eval_X, eval_y = unpickle(file + '5')

    return train_X, train_y, eval_X, eval_y


# 模型定义
def cnn_model_fn(features, labels, mode):
    # 输入层 32x32 ,通道3
    input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
    # 第1个卷积层 卷积核5x5，激活函数ReLU
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=48,
        kernel_size=[5, 5],
        padding='VALID',
        activation=tf.nn.relu
    )
    # 第一个汇合层 大小2x2
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=1
    )
    # 第2个卷积层 卷积核5x5，激活函数ReLU
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding='VALID',
        activation=tf.nn.relu
    )
    # 第2个汇合层 大小2x2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )
    # 第3个卷积层 卷积核3x3，激活函数ReLU
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # 第4个卷积层 卷积核3x3，激活函数ReLU
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # 第5个卷积层 卷积核3x3，激活函数ReLU
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # 第3个汇合层 大小2x2
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=1
    )

    pool3_flat = tf.reshape(pool3, [-1, 11 * 11 * 128])
    # 全连接层FC1
    dense1 = tf.layers.dense(pool3_flat, units=1024, activation=tf.nn.relu)
    # dropout1
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全连接层FC2
    dense2 = tf.layers.dense(dropout1, units=1024, activation=tf.nn.relu)
    # dropout2
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 输出层
    logits = tf.layers.dense(inputs=dropout2, units=10)

    # 预测结果
    predictions = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {
        # 'accuracy': tf.metrics.accuracy(
        #     labels=labels, predictions=predictions)}
        'accuracy': tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


# 模型训练
def train():
    train_X, train_y, eval_X, eval_y = train_test_split('G:/dataset/cifar-10-batches/data_batch_')
    train_data = np.array(train_X, dtype=np.float32)
    train_labels = np.array(train_y, dtype=np.int32)
    eval_data = np.array(eval_X, dtype=np.float32)
    eval_labels = np.array(eval_y, dtype=np.int32)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='/tmp/cifar_alexnet_model')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        # hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print('-----------------evaluate error rate------------------')
    print(eval_results)


def test():
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='/tmp/cifar_alexnet_model')

    test_images, labels = unpickle('G:/dataset/cifar-10-batches/test_batch')
    eval_data = np.array(test_images, dtype=np.float32)
    eval_labels = np.array(labels, dtype=np.int32)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print('-----------------test error rate------------------')
    print(eval_results)


# 预测
def predict():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='/tmp/cifar_alexnet_model')

    test_images, labels = unpickle('G:/dataset/cifar-10-batches/test_batch')
    test_data = np.array(test_images, dtype=np.float32)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        num_epochs=1,
        shuffle=False)
    predictions = mnist_classifier.predict(input_fn=test_input_fn)
    pre = [k for k in predictions]


def main(argv):
    # train()
    test()


if __name__ == '__main__':
    tf.app.run()
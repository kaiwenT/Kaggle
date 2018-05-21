import tensorflow as tf
import tools
import numpy as np
from sklearn import model_selection


tf.logging.set_verbosity(tf.logging.INFO)


# 模型定义
def cnn_model_fn(features, labels, mode):
    # 输入层 28x28
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    # 第一个卷积层 卷积核3x3，激活函数ReLU
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[3, 3],
        padding='VALID',
        activation=tf.nn.relu
    )

    # 第一个汇合层 大小2x2
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    # 第二个卷积层 卷积核5x5，激活函数ReLU
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[4, 4],
        padding='VALID',
        activation=tf.nn.relu
    )

    # 第二个汇合层 大小2x2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=120,
        kernel_size=[5, 5],
        padding='VALID',
        activation=tf.nn.relu
    )
    # 全连接层

    conv3_flat = tf.reshape(conv3, [-1, 120])
    dense = tf.layers.dense(conv3_flat, units=84, activation=tf.nn.relu)
    # dropout
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # 输出层
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions['classes'])

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


# 模型训练
def train():
    images, labels = tools.readtrainset('G:/Kaggle/DigitRecognizer/train.csv')
    X, X1, y, y1 = model_selection.train_test_split(images, labels, test_size=0.2)
    train_data = np.array(X, dtype=np.float32)
    train_labels = np.array(y, dtype=np.int32)
    eval_data = np.array(X1, dtype=np.float32)
    eval_labels = np.array(y1, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='/tmp/mnist_lenet5_model')

    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


# 做预测
def predict():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='/tmp/mnist_lenet5_model')

    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    test_images = tools.readtestset('G:/Kaggle/DigitRecognizer/test.csv')
    test_data = np.array(test_images, dtype=np.float32)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        num_epochs=1,
        shuffle=False)
    predictions = mnist_classifier.predict(input_fn=test_input_fn)
    pre = [k for k in predictions]

    np.savetxt('G:/Kaggle/DigitRecognizer/lenet5_submission.csv',
               np.c_[range(1, 28001), pre],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')


def main(argv):
    train()
    predict()


if __name__ == '__main__':
    tf.app.run()
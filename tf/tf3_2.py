import tensorflow as tf
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y=tf.matmul(x,w)
print y
with tf.Session() as sess:
    print sess.run(y)



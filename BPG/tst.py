import numpy as np
import tensorflow as tf

obs = tf.placeholder(tf.float32, [3, 3], name="obs")
act = tf.placeholder(tf.float32, [3, 1], name="act")
value = tf.placeholder(tf.float32, [3, 1], name="value")

#cast_act = tf.one_hot(act, 2)

state_act = tf.concat([obs, act], axis=1)

w = tf.get_variable("w", [4, 1])
b = tf.get_variable("b", [1])
predict = tf.matmul(state_act, w)# + b

gradient = tf.gradients(predict, act)

loss = tf.reduce_mean(tf.square(value-predict))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

obs_data = np.asarray([[1., 2., 3.], [4., -1., 9.], [6., 3., 0.]])
act_data = np.asarray([[1], [2], [3]])
value_data = np.asarray([[1], [2], [0]])

for i in range(100):
    w_, b_, pred, grad, _ = sess.run([w, b, predict, gradient, train_step], feed_dict={obs: obs_data, act: act_data, value: value_data})
    print("------------------------")
    print("w: ", w_)
    print("b: ", b_)
    print("predict: ", pred)
    print("gradient: ", grad)
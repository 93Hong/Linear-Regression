import tensorflow as tf

tf.set_random_seed(777)

x_train = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, w_val, b_val, _ = \
		sess.run([cost, w, b, train],
			feed_dict = {x_train: [1, 2, 3, 4, 5],
				y_train: [2.1, 3.1, 4.1, 5.1, 6.1]})
	if step % 20 == 0:
		print(step, cost_val, w_val, b_val)

print(sess.run(hypothesis, feed_dict = {x_train: [5]}))

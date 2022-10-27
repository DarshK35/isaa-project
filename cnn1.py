from tensorflow import compat as cm
import numpy as np # numpy module
import os # path join


DATA_DIR = "D:\\Darsh\\Work\\VIT\\Sem 5 - Fall sem 2022-23\\CSE3501 ISAA\\Project\\Dataset256\\"
TRAINING_SET_SIZE = 5812
BATCH_SIZE = 10
IMAGE_SIZE = 224


def _int64_feature(value):
	return cm.v1.train.Feature(int64_list=cm.v1.train.Int64List(value=value))

def _bytes_feature(value):
	return cm.v1.train.Feature(bytes_list=cm.v1.train.BytesList(value=[value]))

# image object from protobuf
class _image_object:
	def __init__(self):
		self.image = cm.v1.Variable([], dtype = cm.v1.string)
		self.height = cm.v1.Variable([], dtype = cm.v1.int64)
		self.width = cm.v1.Variable([], dtype = cm.v1.int64)
		self.filename = cm.v1.Variable([], dtype = cm.v1.string)
		self.label = cm.v1.Variable([], dtype = cm.v1.int32)

## extracting information and storing them in an image object.
def read_and_decode(filename_queue):
	reader = cm.v1.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = cm.v1.parse_single_example(serialized_example, features = {
		"image/encoded": cm.v1.FixedLenFeature([], cm.v1.string),
		"image/height": cm.v1.FixedLenFeature([], cm.v1.int64),
		"image/width": cm.v1.FixedLenFeature([], cm.v1.int64),
		"image/filename": cm.v1.FixedLenFeature([], cm.v1.string),
		"image/class/label": cm.v1.FixedLenFeature([], cm.v1.int64),})
	image_encoded = features["image/encoded"]
	image_raw = cm.v1.image.decode_jpeg(image_encoded, channels=3)
	image_object = _image_object()
	image_object.image = cm.v1.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
	image_object.height = features["image/height"]
	image_object.width = features["image/width"]
	image_object.filename = features["image/filename"]
	image_object.label = cm.v1.cast(features["image/class/label"], cm.v1.int64)
	return image_object

## read input images for training and testing from .tfrecord files.
def malware_input(if_random = True, if_training = True):
	if(if_training):
		filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)]
	else:
		filenames = [os.path.join(DATA_DIR, "eval-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)] ## changing eval to train here!!

	for f in filenames:
		if not cm.v1.gfile.Exists(f):
			raise ValueError("Failed to find file: " + f)
	filename_queue = cm.v1.train.string_input_producer(filenames)
	image_object = read_and_decode(filename_queue)
	image = cm.v1.image.per_image_standardization(image_object.image)
#    image = image_object.image
#    image = cm.v1.image.adjust_gamma(cm.v1.cast(image_object.image, cm.v1.float32), gamma=1, gain=1) # Scale image to (0, 1)
	label = image_object.label
	filename = image_object.filename

	if(if_random):
		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
		print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
		num_preprocess_threads = 1
		image_batch, label_batch, filename_batch = cm.v1.train.shuffle_batch(
			[image, label, filename],
			batch_size = BATCH_SIZE,
			num_threads = num_preprocess_threads,
			capacity = min_queue_examples + 3 * BATCH_SIZE,
			min_after_dequeue = min_queue_examples)
		return image_batch, label_batch, filename_batch
	else:
		image_batch, label_batch, filename_batch = cm.v1.train.batch(
			[image, label, filename],
			batch_size = BATCH_SIZE,
			num_threads = 1)
		return image_batch, label_batch, filename_batch


def weight_variable(shape):
	initial = cm.v1.truncated_normal(shape, stddev=0.05)
	return cm.v1.Variable(initial)

def bias_variable(shape):
	initial = cm.v1.constant(0.02, shape=shape)
	return cm.v1.Variable(initial)

def conv2d(x, W):
	return cm.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return cm.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## constructing the network.
def malware_inference(image_batch):
	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])

	x_image = cm.v1.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

	h_conv1 = cm.v1.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1) # 112

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = cm.v1.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2) # 56

	W_conv3 = weight_variable([5, 5, 64, 128])
	b_conv3 = bias_variable([128])

	h_conv3 = cm.v1.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3) # 28

	W_conv4 = weight_variable([5, 5, 128, 256])
	b_conv4 = bias_variable([256])

	h_conv4 = cm.v1.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
	h_pool4 = max_pool_2x2(h_conv4) # 14

	W_conv5 = weight_variable([5, 5, 256, 256])
	b_conv5 = bias_variable([256])

	h_conv5 = cm.v1.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
	h_pool5 = max_pool_2x2(h_conv5) # 7

	# with cm.v1.variable_scope("fc"):
	W_fc1 = weight_variable([7*7*256, 2048])
	b_fc1 = bias_variable([2048])

	h_pool5_flat = cm.v1.reshape(h_pool5, [-1, 7*7*256])
	h_fc1 = cm.v1.nn.relu(cm.v1.matmul(h_pool5_flat, W_fc1) + b_fc1)

	h_fc1_drop = cm.v1.nn.dropout(h_fc1, 1.0)

	W_fc2 = weight_variable([2048, 256])
	b_fc2 = bias_variable([256])

	h_fc2 = cm.v1.nn.relu(cm.v1.matmul(h_fc1_drop, W_fc2) + b_fc2)

	W_fc3 = weight_variable([256, 64])
	b_fc3 = bias_variable([64])

	h_fc3 = cm.v1.nn.relu(cm.v1.matmul(h_fc2, W_fc3) + b_fc3)

	W_fc4 = weight_variable([64, 5])
	b_fc4 = bias_variable([5])

	y_conv = cm.v1.nn.softmax(cm.v1.matmul(h_fc3, W_fc4) + b_fc4) 
#    y_conv = cm.v1.matmul(h_fc3, W_fc4) + b_fc4

	return y_conv


def malware_train():
	image_batch_out, label_batch_out, filename_batch = malware_input(if_random = False, if_training = True)

	image_batch_placeholder = cm.v1.placeholder(cm.v1.float32, shape=[BATCH_SIZE, 224, 224, 3])
	image_batch = cm.v1.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

	label_batch_placeholder = cm.v1.placeholder(cm.v1.float32, shape=[BATCH_SIZE, 5])
	label_offset = -cm.v1.ones([BATCH_SIZE], dtype=cm.v1.int64, name="label_batch_offset")
	label_batch_one_hot = cm.v1.one_hot(cm.v1.add(label_batch_out, label_offset), depth=5, on_value=1.0, off_value=0.0)

	logits_out = malware_inference(image_batch_placeholder)
	loss = cm.v1.reduce_sum(cm.v1.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
	# loss = cm.v1.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

	train_step = cm.v1.train.GradientDescentOptimizer(0.007).minimize(loss) ## 0.007 is the learning rate.

	saver = cm.v1.train.Saver()

	with cm.v1.Session() as sess:
		# Visualize the graph through tensorboard.
		file_writer = cm.v1.summary.FileWriter("./logs", sess.graph)

		sess.run(cm.v1.global_variables_initializer())
		# saver.restore(sess, "C:/Users/admin/Project IAS/checkpoint-train.ckpt")
		coord = cm.v1.train.Coordinator() ## A coordinator for threads implements a mechanism to coordinate the termination of a set of threads.
		threads = cm.v1.train.start_queue_runners(coord=coord, sess = sess) ## Start all the queue runners collected in the graph.

		for i in range(TRAINING_SET_SIZE ):#* 100):
			image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

			_, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

			print(i)
			print(image_out.shape)
			print("label_out: ")
			print(filename_out)
			print(label_out)
			print(label_batch_one_hot_out)
			print("infer_out: ")
			print(infer_out)
			print("loss: ")
			print(loss_out)
			if(i%100 == 0): ## change to 50
				saver.save(sess, "D:\\Darsh\\Work\\VIT\\Sem 5 - Fall sem 2022-23\\CSE3501 ISAA\\Project\\Checkpoints\\checkpoint.ckpt")

		coord.request_stop()
		coord.join(threads)
		sess.close()



def malware_eval():
	image_batch_out, label_batch_out, filename_batch = malware_input(if_random = False, if_training = False)

	image_batch_placeholder = cm.v1.placeholder(cm.v1.float32, shape=[BATCH_SIZE, 224, 224, 3])
	image_batch = cm.v1.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

	label_tensor_placeholder = cm.v1.placeholder(cm.v1.int64, shape=[BATCH_SIZE])
	label_offset = -cm.v1.ones([BATCH_SIZE], dtype=cm.v1.int64, name="label_batch_offset") ## cm.v1.ones create a tensor mentioned shape(1st arg) and type(2nd arg) of 1's.
	label_batch = cm.v1.add(label_batch_out, label_offset)

	logits_out = cm.v1.reshape(malware_inference(image_batch_placeholder), [BATCH_SIZE, 5])
	logits_batch = cm.v1.to_int64(cm.v1.arg_max(logits_out, dimension = 1))

	correct_prediction = cm.v1.equal(logits_batch, label_tensor_placeholder)
	accuracy = cm.v1.reduce_mean(cm.v1.cast(correct_prediction, cm.v1.float32))

	saver = cm.v1.train.Saver()

	with cm.v1.Session() as sess:
		sess.run(cm.v1.global_variables_initializer())
		saver.restore(sess, "D:\\Darsh\\Work\\VIT\\Sem 5 - Fall sem 2022-23\\CSE3501 ISAA\\Project\\Checkpoints\\checkpoint.ckpt")
		coord = cm.v1.train.Coordinator() ## A coordinator for threads implements a mechanism to coordinate the termination of a set of threads.
		threads = cm.v1.train.start_queue_runners(coord=coord, sess = sess)

		accuracy_accu = 0.0

		for i in range(30):
			image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

			accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
			accuracy_accu += accuracy_out

			print(i)
			# print(image_out.shape)
			# print("label_out: ")
			# print(filename_out)
			# print(label_out)
			# print(logits_batch_out)
			# print (accuracy_accu, accuracy_out)

		print("Accuracy: ")
		print(accuracy_accu/30)

		coord.request_stop()
		coord.join(threads)
		sess.close()

# malware_train()
malware_eval()
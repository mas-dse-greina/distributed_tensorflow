
# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to distribute over
# Then run this script on each of those machines.

"""
Usage:  python test_dist.py --ip=10.100.68.245 --is_sync=0
		for asynchronous TF
		python test_dist.py --ip=10.100.68.245 --is_sync=1
		for synchronous updates
		The IP address must match one of the ones in the list below. If not passed,
		then we"ll default to the current machine"s IP (which is usually correct unless you use OPA)
"""
ps_hosts = ["10.100.68.245"]
ps_ports = ["2222"]
worker_hosts = ["10.100.68.193","10.100.68.183"] #,"10.100.68.185","10.100.68.187"]
worker_ports = ["2222", "2222"] #, "2222", "2222"]

ps_list = ["{}:{}".format(x,y) for x,y in zip(ps_hosts, ps_ports)]
worker_list = ["{}:{}".format(x,y) for x,y in zip(worker_hosts, worker_ports)]
print ("Distributed TensorFlow training")
print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))

# Tensorflow is trying to fit a line with random noise added.
# So this is a standard regression with Distributed TF
slope = 5
intercept = 13

CHECKPOINT_DIRECTORY = "./checkpoints/"
NUM_STEPS = 50000

####################################################################

import numpy as np
import tensorflow as tf
import os
import socket

# Unset proxy env variable to avoid gRPC errors
del os.environ["http_proxy"]
del os.environ["https_proxy"]

# You can turn on the gRPC messages by setting the environment variables below
#os.environ["GRPC_VERBOSITY"]="DEBUG"
#os.environ["GRPC_TRACE"] = "all"

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # Get rid of the AVX, SSE warnings

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate.")
tf.app.flags.DEFINE_integer("steps_to_validate", 1000,
					 "Validate and print loss after this many steps")
tf.app.flags.DEFINE_integer("is_sync", 0, "Synchronous updates?")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()), "IP address of this machine")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


if (FLAGS.ip in ps_hosts):
	job_name = "ps"
	task_index = ps_hosts.index(FLAGS.ip)
elif (FLAGS.ip in worker_hosts):
	job_name = "worker"
	task_index = worker_hosts.index(FLAGS.ip)
else:
	print("Error: IP {} not found in the worker or ps node list.\nUse --ip= to specify which machine this is.".format(FLAGS.ip))
	exit()

def create_done_queue(i):
  """
  Queue used to signal termination of the i"th ps shard. 
  Each worker sets their queue value to 1 when done.
  The parameter server op just checks for this.
  """
  
  with tf.device("/job:ps/task:{}".format(i)):
	return tf.FIFOQueue(len(worker_hosts), tf.int32, 
		shared_name="done_queue{}".format(i))
  
def create_done_queues():
  return [create_done_queue(i) for i in range(len(ps_hosts))]

def loss(label, pred):
  return tf.losses.mean_squared_error(label, pred)


def main(_):

  cluster = tf.train.ClusterSpec({"ps": ps_list, "worker": worker_list})
  server = tf.train.Server(cluster,job_name=job_name,task_index=task_index)

  is_sync = (FLAGS.is_sync == 1)  # Synchronous or asynchronous updates
  is_chief = (task_index == 0)  # Am I the chief node (always task 0)


  if job_name == "ps":

	sess = tf.Session(server.target)
	queue = create_done_queue(task_index)

	print("\n")
	print("*"*30)
	print("\nParameter server #{} on this machine.\n\n" \
		"Waiting on workers to finish.\n\nPress CTRL-\\ to terminate early." .format(task_index))
	print("*"*30)

	# wait until all workers are done
	for i in range(len(worker_hosts)):
		sess.run(queue.dequeue())
		print("Worker #{} reports job finished." .format(i))
	 
	print("Parameter server #{} is quitting".format(task_index))
	print("Training complete.")

  elif job_name == "worker":
	
	if is_chief:
		print("I am chief worker {} with task #{}".format(worker_hosts[task_index], task_index))
	else:
		print("I am worker {} with task #{}".format(worker_hosts[task_index], task_index))

	with tf.device(tf.train.replica_device_setter(
					worker_device="/job:worker/task:{}".format(task_index),
					cluster=cluster)):
	  global_step = tf.Variable(0, name="global_step", trainable=False)

	  """
	  BEGIN: Define our model
	  """
	  inputv = tf.placeholder(tf.float32)
	  label  = tf.placeholder(tf.float32)

	  # NOTE: The [] means a scalar value. It is different than [1] (which is a vector of length 1)
	  weight = tf.get_variable("slope", [], tf.float32, initializer=tf.random_normal_initializer())
	  bias  = tf.get_variable("intercept", [], tf.float32, initializer=tf.random_normal_initializer())
	  pred = tf.multiply(inputv, weight) + bias

	  loss_value = loss(label, pred)
	  """
	  END: Define our model
	  """

	  # Define gradient descent optimizer
	  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	  grads_and_vars = optimizer.compute_gradients(loss_value)
	  if is_sync:
		
		rep_op = tf.train.SyncReplicasOptimizer(optimizer,
			replicas_to_aggregate=len(worker_hosts),
			total_num_replicas=len(worker_hosts),
			use_locking=True)

		train_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)

		init_token_op = rep_op.get_init_tokens_op()

		chief_queue_runner = rep_op.get_chief_queue_runner()

	  else:
		
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


	  init_op = tf.global_variables_initializer()
	  
	  saver = tf.train.Saver()

	  # These are the values we wish to print to TensorBoard
	  tf.summary.scalar("slope", weight)
	  tf.summary.scalar("intercept", bias)
	  tf.summary.scalar("loss", loss_value)
	  tf.summary.histogram("loss", loss_value)
	  tf.summary.histogram("slope", weight)
	  tf.summary.histogram("intercept", bias)
	  
	# Need to remove the checkpoint directory before each new run
	import shutil
	shutil.rmtree(CHECKPOINT_DIRECTORY, ignore_errors=True)

	# Send a signal to the ps when done by simply updating a queue in the shared graph
	enq_ops = []
	for q in create_done_queues():
		qop = q.enqueue(1)
		enq_ops.append(qop)

	summary_op = tf.summary.merge_all()
	# if is_chief:
	# 	summary_op = tf.summary.merge_all()
	# else:
	# 	summary_op = None

	# TODO:  Theoretically I can pass the summary_op into
	# the Supervisor and have it handle the TensorBoard
	# log entries. However, doing so seems to hang the code.
	# For now, I just handle the summary calls explicitly.
	sv = tf.train.Supervisor(is_chief=is_chief,
		logdir=CHECKPOINT_DIRECTORY,
		init_op=init_op,
		summary_op=None, 
		saver=saver,
		global_step=global_step,
		save_model_secs=60)  # Save the model (with weights) everty 60 seconds

	# TODO:
	# I'd like to use managed_session for this as it is more abstract
	# and probably less sensitive to changes from the TF team. However,
	# I am finding that the chief worker hangs on exit if I use managed_session.
	with sv.prepare_or_wait_for_session(server.target) as sess:
	#with sv.managed_session(server.target) as sess:
	
	  
		if is_chief and is_sync:
			sv.start_queue_runners(sess, [chief_queue_runner])
			sess.run(init_token_op)

		step = 0

		while (not sv.should_stop()) and (step < NUM_STEPS):

			# Define a line with random noise
			train_x = np.random.randn(1)*10
			train_y = slope * train_x  + intercept + np.random.randn(1) * 0.33

			history, loss_v, step = sess.run([train_op, loss_value, global_step], 
										feed_dict={inputv:train_x, label:train_y})
		
			if (step % steps_to_validate == 0):
			  w,b = sess.run([weight,bias])
			  
			  print("[step: {:,} of {:,}] Predicted Slope: {:.3f} (True slope = {}), " \
					"Predicted Intercept: {:.3f} (True intercept = {}), loss: {:.4f}" \
					.format(step, NUM_STEPS, w, slope, b, intercept, loss_v))

			  summary = sess.run(summary_op, feed_dict={inputv:train_x, label:train_y})
			  sv.summary_computed(sess, summary)  # Update the summary

	  
		 # Send a signal to the ps when done by simply updating a queue in the shared graph
		for op in enq_ops:
			sess.run(op)   # Send the "work completed" signal to the parameter server
			
				
	print('Finished work on this node.')
	sv.request_stop()
	#sv.stop()


if __name__ == "__main__":
  tf.app.run()




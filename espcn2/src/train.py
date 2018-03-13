import tensorflow as tf
import os
import time
from PIL import Image

from data_provider import read_data
from networks import espcn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 150000, "Number of epoch")
tf.app.flags.DEFINE_integer("image_size", 17, "The size of image input")
tf.app.flags.DEFINE_boolean("is_train", True, "The size of image input")
tf.app.flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
tf.app.flags.DEFINE_integer("c_dim", 3, "The size of channel")
tf.app.flags.DEFINE_integer("batch_size", 128, "the size of batch")
#tf.app.flags.DEFINE_string("test_img", "C:\\image_path\\1343240909729.jpg", "test_img")
tf.app.flags.DEFINE_float("learning_rate", 1e-5 , "The learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "C:\\Users\\Administrator\\git\\espcn2\\espcn2\\checkpoint", "Name of checkpoint directory")
tf.app.flags.DEFINE_string("inputdata", 'C:\\Users\\Administrator\\git\\espcn2\\espcn2\\outdata\\sr-data.tfrecord', "outdata")
#789544
def main(_):
  with tf.Session() as sess:
    
    
    with tf.name_scope('inputs'), tf.device('/cpu:0'):
      input_, label_, iterator_ = read_data(FLAGS)
      
    normal_input=tf.divide(input_, 255)
    label_=tf.divide(label_, 255)
    ps=espcn(FLAGS,normal_input)
    
    loss = tf.reduce_mean(tf.squared_difference(ps,label_))
    sr_images=tf.multiply(ps, 255)
    label_=tf.multiply(label_, 255)
    
    input_ = tf.image.resize_images(input_, (FLAGS.image_size*FLAGS.scale, FLAGS.image_size*FLAGS.scale))
    
    summary_images = tf.concat([input_,label_, sr_images], axis=1)      
    tf.summary.image('images', summary_images, max_outputs=4)
    
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    
    counter = 0
    time_ = time.time()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    load(saver,sess)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, sess.graph)
    sess.run(iterator_.initializer)
    while True:
      counter += 1
      _, err = sess.run([train_op, loss])
      if counter % 10 == 0:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary_str,_ = sess.run([summary_op,train_op],options=run_options,run_metadata=run_metadata)
          summary_writer.add_run_metadata(run_metadata, 'step%03d' % counter)
          summary_writer.add_summary(summary_str, counter)
          print("step: [%2d], time: [%4.4f], loss: [%.8f]" % (counter, time.time()-time_, err))
      if counter % 500 == 0:
          save(saver,sess, counter)

def load(saver,sess):
  """
      To load the checkpoint use to test or pretrain
  """
  print("\nReading Checkpoints.....\n\n")
  model_dir = "%s_%s_%s" % ("espcn", FLAGS.image_size,FLAGS.scale)# give the model name by label_size
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  
  # Check the checkpoint is exist 
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
    saver.restore(sess, os.path.join(os.getcwd(), ckpt_path))
    print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
  else:
    print("\n! Checkpoint Loading Failed \n\n")
    
def save(saver,sess,step):
  """
      To save the checkpoint use to test or pretrain
  """
  model_name = "ESPCN.model"
  model_dir = "%s_%s_%s" % ("espcn", FLAGS.image_size,FLAGS.scale)
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
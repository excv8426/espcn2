import tensorflow as tf
import numpy as np
import os

from networks import espcn
from PIL import Image


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("image_size", 17, "The size of image input")
tf.app.flags.DEFINE_boolean("is_train", False, "The size of image input")
tf.app.flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
tf.app.flags.DEFINE_integer("c_dim", 3, "The size of channel")
tf.app.flags.DEFINE_string("test_img", "C:\\image_path\\1343240909729.jpg", "test_img")
tf.app.flags.DEFINE_string("checkpoint_dir", "C:\\Users\\Administrator\\git\\espcn2\\espcn2\\checkpoint", "Name of checkpoint directory")

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
    
def main(_):
  with tf.Session() as sess:    
    input_image = Image.open(FLAGS.test_img)
    image_placeholder = tf.placeholder(tf.float32, shape=[input_image.height,input_image.width,3])
    input_ = tf.expand_dims(image_placeholder, 0)
    
    normal_input=tf.divide(input_, 255)
    
    ps=espcn(FLAGS,normal_input)
    sr_images=tf.multiply(ps, 255)
    
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    load(saver,sess)
    #threads = tf.train.start_queue_runners(sess)
    #sr_image=sess.run(sr_images)
    sr_image=sess.run(sr_images,feed_dict={
      image_placeholder:input_image})
    result_image = Image.fromarray(np.uint8(sr_image.reshape(sr_image.shape[1:4])), mode='RGB')
    result_image.convert('RGB').save('C:\\image_path\\espn2.png', 'PNG')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
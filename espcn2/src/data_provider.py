import tensorflow as tf

def read_data(config):
  dataset = tf.data.TFRecordDataset([config.inputdata])

  def parser(record):
      keys_to_features = {
          "image": tf.FixedLenFeature((), tf.string, default_value=""),
          "label": tf.FixedLenFeature((), tf.int64,default_value=tf.zeros([], dtype=tf.int64)),
      }
      parsed = tf.parse_single_example(record, keys_to_features)
  
      image = tf.decode_raw(parsed["image"],tf.uint8)
      image = tf.reshape(image, [config.image_size*config.scale, config.image_size*config.scale, 3])
      label = tf.cast(parsed["label"], tf.int32)
  
      return {"image": image}, label
  
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(config.batch_size)
  dataset = dataset.repeat(config.epoch)
  iterator = dataset.make_initializable_iterator()
  
  features, labels = iterator.get_next()
  hr_images = features['image']
  lr_images = tf.image.resize_images(hr_images, (config.image_size, config.image_size))
  
  lr_images=tf.cast(lr_images, tf.float32)
  hr_images=tf.cast(hr_images, tf.float32)
  return lr_images,hr_images,iterator
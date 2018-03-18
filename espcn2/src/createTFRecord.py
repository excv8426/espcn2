import os
import tensorflow as tf
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_dir', 'D:\\123456', "学習画像のディレクトリ")
tf.app.flags.DEFINE_integer("image_size", 17, "The size of image input")
tf.app.flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
tf.app.flags.DEFINE_integer("mini_batch", 1000, "mini_batch")
tf.app.flags.DEFINE_string("outdata", 'D:\\sr-data.tfrecord', "outdata")

def createTFRecord():
    writer = tf.python_io.TFRecordWriter(FLAGS.outdata)
    
    with tf.Session() as sess:
        image_files = os.walk(FLAGS.image_dir)
        file_list=[]
        for root, dirs, files in image_files:
            for file in files:
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))
                
        filename_queue = tf.train.string_input_producer(file_list,shuffle=True)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image = tf.image.decode_jpeg(value, channels=3)
        #imageSize=image.shape[0]*image.shape[1]
        #print(imageSize)
        #miniBatch=imageSize/FLAGS.image_size/FLAGS.scale
        
        images=[]
        for i in range(FLAGS.mini_batch):
            images.append(tf.random_crop(image, (FLAGS.image_size*FLAGS.scale, FLAGS.image_size*FLAGS.scale, 3)))
        #images = tf.stack(images, axis=0)
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(len(file_list)):
            print('i',i)
            loaded_images=sess.run(images)
            for loaded_image in loaded_images:                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[loaded_image.tobytes()]))
                    }))
                writer.write(example.SerializeToString())
            #print(sess.run(image).shape)
            #print(sess.run(images)[4].shape)
            
        writer.close()
        


def readTFRecord():
    filename_queue=tf.train.string_input_producer([FLAGS.outdata])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image' : tf.FixedLenFeature([], tf.string),
        })
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [FLAGS.image_size, FLAGS.image_size, 3])
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        while True:
            print(sess.run(img).shape)
            print(sess.run(img))
    
    
    
def dataset_input_fn():
    filenames = [FLAGS.outdata]
    dataset = tf.data.TFRecordDataset(filenames)
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
    
        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["image"],tf.uint8)
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 3])
        label = tf.cast(parsed["label"], tf.int32)
    
        return {"image": image}, label
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(5)
    dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    
    features, labels = iterator.get_next()
    img = features['image']
    lr_images = tf.image.resize_images(img, (5, 5))

    with tf.Session() as sess:
        #print(sess.run(img))
        
        sess.run(iterator.initializer)
        while True:
            print(sess.run(lr_images).shape)
    return features, labels
        
    
 
def main(argv=None):
    #dataset_input_fn()
    createTFRecord()
    #readTFRecord()
    

if __name__ == '__main__':
    tf.app.run()
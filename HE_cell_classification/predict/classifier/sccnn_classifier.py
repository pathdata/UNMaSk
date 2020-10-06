import tensorflow as tf

from classifier.subpackages import inference
from classifier.subpackages import generate_output, run_training, loss_function


class SccnnClassifier:
    def __init__(self, batch_size, image_height, image_width, in_feat_dim,
                 in_label_dim, num_of_classes, tf_device=None):
        if tf_device is None:
            tf_device = ['/gpu:0']
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.in_feat_dim = in_feat_dim
        self.num_of_classes = num_of_classes
        self.in_label_dim = in_label_dim
        self.loss = None
        self.accuracy = None
        self.logits = None
        self.tf_device = tf_device
        self.LearningRate = None
        # for d in self.tf_device:
        #     with tf.device(d):
        self.images = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.image_height,
                                            self.image_width, self.in_feat_dim])
        self.labels = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.in_label_dim])

    def run_checks(self, opts):
        assert (opts.image_height == self.image_height)
        assert (opts.image_width == self.image_width)
        assert (opts.in_feat_dim == self.in_feat_dim)
        assert (opts.in_label_dim == self.in_label_dim)
        return 0

    def run_training(self, opts):
        network = run_training.run_training(network=self, opts=opts)

        return network

    def generate_output(self, opts, network_output=True):

        generate_output.generate_output(network=self, opts=opts,
                                        network_output=network_output
                                        )

    def generate_output_sub_dir(self, opts, sub_dir_name, network_output=True):
        output_path = generate_output.generate_output_sub_dir(network=self, opts=opts,
                                                              sub_dir_name=sub_dir_name,
                                                              network_output=network_output
                                                              )
        print('Output Files saved at:' + output_path)

    def inference(self, is_training):

        self.logits, output = inference.inference(network=self,
                                                  is_training=is_training)

        return self.logits, output

    def loss_function(self, weighted_loss_per_class=None):

        self.loss = loss_function.loss_function(network=self,
                                                weighted_loss_per_class=weighted_loss_per_class)

        return self.loss

    def train(self):
        loss = self.loss
        with tf.name_scope('Optimization'):
            train_op = tf.train.AdagradOptimizer(learning_rate=self.LearningRate).minimize(loss)

        return train_op

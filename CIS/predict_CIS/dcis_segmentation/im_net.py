import tensorflow as tf

from dcis_segmentation.subpackages import random_crop, loss_function, inference, run_training, generate_output


class IMNet:
    def __init__(self, batch_size, image_height, image_width, in_feat_dim,
                 label_height, label_width, in_label_dim,
                 num_of_classes=2, crop_height=None, crop_width=None, tf_device=None):
        if crop_height is None:
            crop_height = 508
        if crop_width is None:
            crop_width = 508
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
        self.logits_b1 = None
        self.logits_b2 = None
        self.logits_b3 = None
        self.tf_device = tf_device
        self.LearningRate = None
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.label_height = label_height
        self.label_width = label_width
        # for d in self.tf_device:
        #     with tf.device(d):
        self.images = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.image_height,
                                            self.image_width, self.in_feat_dim])
        self.labels = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.label_height,
                                            self.label_width, self.in_label_dim])

    def run_checks(self, opts):
        assert (opts.image_height == self.image_height)
        assert (opts.image_width == self.image_width)
        assert (opts.in_feat_dim == self.in_feat_dim)
        assert (opts.label_dim == self.in_label_dim)
        assert (opts.num_of_classes == self.num_of_classes)
        return 0

    def random_crop(self, images=None, labels=None):
        if images is None:
            images = self.images

        if labels is None:
            labels = images.labels

        images, labels = random_crop.random_crop(opts=self, images=images, labels=labels)

        return images, labels

    def inference(self, is_training, images=None):
        if images is None:
            images = self.images

        self.logits, self.logits_b1, self.logits_b2, self.logits_b3 = inference.inference(network=self,
                                                                                          is_training=is_training,
                                                                                          images=images)
        return self.logits, self.logits_b1, self.logits_b2, self.logits_b3

    def run_training(self, opts):
        network = run_training.run_training(network=self, opts=opts)

        return network

    def generate_output(self, opts, save_pre_process=True, network_output=True, post_process=True):

        generate_output.generate_output(network=self,
                                        opts=opts,
                                        save_pre_process=save_pre_process,
                                        network_output=network_output,
                                        post_process=post_process)

    def generate_output_sub_dir(self, opts, sub_dir_name, save_pre_process=True, network_output=True,
                                post_process=True):
        output_path = generate_output.generate_output_sub_dir(network=self,
                                                              opts=opts,
                                                              sub_dir_name=sub_dir_name,
                                                              save_pre_process=save_pre_process,
                                                              network_output=network_output,
                                                              post_process=post_process)
        print('Output Files saved at:' + output_path)

    def loss_function(self,
                      logits=None,
                      logits_b1=None,
                      logits_b2=None,
                      logits_b3=None,
                      labels=None,
                      global_step=None):

        if logits is None:
            logits = self.logits

        if logits_b1 is None:
            logits_b1 = self.logits_b1

        if logits_b2 is None:
            logits_b2 = self.logits_b2

        if logits_b3 is None:
            logits_b3 = self.logits_b3

        if labels is None:
            labels = self.labels

        self.loss = loss_function.aux_plus_main_loss(logits=logits,
                                                     logits_b1=logits_b1,
                                                     logits_b2=logits_b2,
                                                     logits_b3=logits_b3,
                                                     labels=labels,
                                                     global_step=global_step)

        return self.loss

    def train(self, loss=None, lr=None):
        if loss is None:
            loss = self.loss
        if lr is None:
            lr = self.LearningRate
        with tf.name_scope('Optimization'):
            train_op = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)

        return train_op

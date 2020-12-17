import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
import time
from datetime import datetime

from dcis_segmentation.subpackages import data_utils, random_crop


def run_training(network, opts):
    param = sio.loadmat(os.path.join(opts.data_dir, opts.train_data_filename + '.mat'))
    train_num_examples = param['num_examples'][0][0]

    param = sio.loadmat(os.path.join(opts.data_dir, opts.valid_data_filename + '.mat'))
    valid_num_examples = param['num_examples'][0][0]

    training_dataset = data_utils.get_data_set(
        filename=os.path.join(opts.data_dir, opts.train_data_filename + '.tfrecords'),
        num_epochs=opts.num_of_epoch,
        shuffle_size=opts.batch_size * 10,
        batch_size=opts.batch_size,
        prefetch_buffer=opts.batch_size * 2)

    validation_dataset = data_utils.get_data_set(
        filename=os.path.join(opts.data_dir, opts.valid_data_filename + '.tfrecords'),
        num_epochs=opts.num_of_epoch,
        shuffle_size=opts.batch_size * 10,
        batch_size=opts.batch_size,
        prefetch_buffer=opts.batch_size * 2)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    data, labels = iterator.get_next()
    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    data, labels = data_utils.augment(data_in=data, labels_in=labels)
    data = data_utils.random_variations(data_in=data)

    global_step = tf.Variable(1.0, trainable=False)
    network.LearningRate = tf.placeholder(tf.float32)

    images, labels = random_crop.random_crop(opts=opts, images=data, labels=labels)
    logits, logits_b1, logits_b2, logits_b3 = network.inference(images=images, is_training=True)

    loss = network.loss_function(logits=logits,
                                 logits_b1=logits_b1,
                                 logits_b2=logits_b2,
                                 logits_b3=logits_b3,
                                 global_step=global_step,
                                 labels=labels)

    train_op = network.train(loss=loss, lr=network.LearningRate)

    imr0 = tf.concat(values=[logits[0:1, :, :, 0:1], logits[0:1, :, :, 1:2]], axis=1)
    imr1 = images[0:1, :, :, :]
    imr2 = tf.concat(values=[labels[0:1, :, :, 0:1], labels[0:1, :, :, 1:2]], axis=1)
    _ = tf.summary.image('Output_1', imr0)
    _ = tf.summary.image('Input_1', imr1)
    _ = tf.summary.image('label', imr2)

    correct_prediction = tf.equal(tf.argmax(labels[:, :, :, 0:2], 3),
                                  tf.argmax(logits[:, :, :, 0:2], 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.summary.scalar('Accuracy', accuracy)

    train_count = int((train_num_examples / opts.batch_size) + 1)
    valid_count = int((valid_num_examples / opts.batch_size) + 1)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    avg_training_loss = 0.0
    avg_validation_loss = 0.0
    avg_training_accuracy = 0.0
    avg_validation_accuracy = 0.0

    training_loss = 0.0
    validation_loss = 0.0
    training_accuracy = 0.0
    validation_accuracy = 0.0

    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'valid'), sess.graph)
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        curr_epoch = opts.current_epoch_num
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            curr_epoch = int(global_step)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
            workspace = sio.loadmat(os.path.join(opts.exp_dir, 'avg_training_loss_acc.mat'))
            avg_training_loss = np.array(workspace['avg_training_loss'])
            avg_validation_loss = np.array(workspace['avg_validation_loss'])
            avg_training_accuracy = np.array(workspace['avg_training_accuracy'])
            avg_validation_accuracy = np.array(workspace['avg_validation_accuracy'])
        else:
            print('No checkpoint file found', flush=True)

        for epoch in range(curr_epoch, opts.num_of_epoch):
            lr = (0.001 * np.exp(-0.01 * ((epoch + 1) - 1))).astype('float32')
            opts.current_epoch_num = global_step
            start_time = time.time()
            sess.run(training_init_op)
            step = 0
            try:
                avg_loss = 0.0
                avg_accuracy = 0.0
                for step in range(train_count):
                    start_time_step = time.time()
                    if (step % int(50) == 0) or (step == 0):
                        images_, labels_, logits_out, summary_str, _, loss_value, accuracy_value = \
                            sess.run([images, labels, logits, summary_op, train_op, loss, accuracy],
                                     feed_dict={network.LearningRate: lr})
                        train_writer.add_summary(summary_str, step + epoch * train_count)
                        inter = {'logits': logits_out,
                                 'input': images_,
                                 'label': labels_}
                        sio.savemat(os.path.join(opts.exp_dir, 'inter_train.mat'), inter)
                        duration = time.time() - start_time_step
                        format_str = (
                            '%s: epoch %d, step %d/ %d, '
                            'Training Loss = %.2f, Training Accuracy = %.2f, (%.2f sec/step)')
                        print(
                            format_str % (
                                datetime.now(), epoch + 1, step + 1, train_count,
                                loss_value, accuracy_value, float(duration)), flush=True)
                    else:
                        _, loss_value, accuracy_value = \
                            sess.run([train_op, loss, accuracy],
                                     feed_dict={network.LearningRate: lr})

                    avg_loss += loss_value
                    avg_accuracy += accuracy_value

                training_loss = avg_loss / train_count
                training_accuracy = avg_accuracy / train_count

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (epoch, step), flush=True)

            sess.run(validation_init_op)
            step = 0
            try:
                avg_loss = 0.0
                avg_accuracy = 0.0
                for step in range(valid_count):
                    start_time_step = time.time()
                    if (step % int(10) == 0) or (step == 0):
                        images_, labels_, logits_out, summary_str, loss_value, accuracy_value = \
                            sess.run([images, labels, logits, summary_op, loss, accuracy])
                        valid_writer.add_summary(summary_str, step + epoch * train_count)
                        inter = {'logits': logits_out,
                                 'input': images_,
                                 'label': labels_}
                        sio.savemat(os.path.join(opts.exp_dir, 'inter_valid.mat'), inter)
                        duration = time.time() - start_time_step
                        format_str = (
                            '%s: epoch %d, step %d/ %d, Validation Loss = %.2f, '
                            'Validation Accuracy = %.2f, (%.2f sec/step)')
                        print(
                            format_str % (
                                datetime.now(), epoch + 1, step + 1, valid_count, loss_value, accuracy_value,
                                float(duration)), flush=True)
                    else:
                        loss_value, accuracy_value = \
                            sess.run([loss, accuracy])

                    avg_loss += loss_value
                    avg_accuracy += accuracy_value

                validation_loss = avg_loss / valid_count
                validation_accuracy = avg_accuracy / valid_count

            except tf.errors.OutOfRangeError:
                print('Done validation for %d epochs, %d steps.' % (epoch, step), flush=True)

            checkpoint_path = os.path.join(opts.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            global_step = global_step + 1
            # Average loss on training and validation datasets.
            duration = time.time() - start_time
            format_str = (
                '%s: epoch %d, Training Loss = %.2f, Validation Loss = %.2f, '
                'Training Accuracy = %.2f, Validation Accuracy = %.2f, (%.2f sec/epoch)')
            print(format_str % (
                datetime.now(), epoch + 1, training_loss, validation_loss,
                training_accuracy, validation_accuracy, float(duration)), flush=True)
            if epoch == 0:
                avg_training_loss = [round(training_loss, 2)]
                avg_validation_loss = [round(validation_loss, 2)]
                avg_training_accuracy = [round(training_accuracy * 100, 2)]
                avg_validation_accuracy = [round(validation_accuracy * 100, 2)]
            else:
                avg_training_loss = np.append(avg_training_loss, [round(training_loss, 2)])
                avg_validation_loss = np.append(avg_validation_loss, [round(validation_loss, 2)])
                avg_training_accuracy = np.append(avg_training_accuracy, [round(training_accuracy * 100, 2)])
                avg_validation_accuracy = np.append(avg_validation_accuracy, [round(validation_accuracy * 100, 2)])
            avg_training_loss_acc_dict = {'avg_training_loss': avg_training_loss,
                                          'avg_validation_loss': avg_validation_loss,
                                          'avg_training_accuracy': avg_training_accuracy,
                                          'avg_validation_accuracy': avg_validation_accuracy,
                                          }
            sio.savemat(file_name=os.path.join(opts.exp_dir, 'avg_training_loss_acc.mat'),
                        mdict=avg_training_loss_acc_dict)
            print(avg_training_loss, flush=True)
            print(avg_validation_loss, flush=True)
            print(avg_training_accuracy, flush=True)
            print(avg_validation_accuracy, flush=True)

        return network

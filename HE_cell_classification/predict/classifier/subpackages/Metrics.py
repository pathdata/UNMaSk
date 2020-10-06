import tensorflow as tf


class Metrics:
    def __init__(self, labels, logits, num_classes):
        self.Labels = labels
        predictions = tf.argmax(logits, 3)
        self.Predictions = predictions[:, :, 0]
        self.num_classes = num_classes

        self.ConfusionMatrix = None
        self.NormalizedConfusionMatrix = None
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.Accuracy = None
        self.AccuracyClass = None
        self.Precision = None
        self.PrecisionClass = None
        self.Sensitivity = None
        self.SensitivityClass = None
        self.Specificity = None
        self.SpecificityClass = None
        self.F1Score = None
        self.F1ScoreClass = None

    def calculate_all(self):
        self.ConfusionMatrix = None

        self.confusion_matrix()
        self.normalized_confusion_matrix()
        self.calculate_metrics()
        self.accuracy()
        self.accuracy_per_class()
        self.precision()
        self.precision_per_class()
        self.sensitivity()
        self.sensitivity_per_class()
        self.specificity()
        self.specificity_per_class()
        self.f1score()
        self.f1score_per_class()

        output = {'Accuracy': self.Accuracy,
                  'AccuracyClass': self.AccuracyClass,
                  'Precision': self.Precision,
                  'PrecisionClass': self.PrecisionClass,
                  'Sensitivity' : self.Sensitivity,
                  'SensitivityClass': self.SensitivityClass,
                  'Specificity': self.Specificity,
                  'SpecificityClass': self.SpecificityClass,
                  'F1Score': self.F1Score,
                  'F1ScoreClass': self.F1ScoreClass,
                  'ConfusionMatrix': self.ConfusionMatrix,
                  'NormalizedConfusionMatrix': self.NormalizedConfusionMatrix
                  }

        return output

    def confusion_matrix(self):
        predictions = tf.squeeze(self.Predictions)
        labels = self.Labels

        label_int = tf.cast(labels - 1, tf.int64)
        self.ConfusionMatrix = tf.confusion_matrix(label_int, predictions)
        return self.ConfusionMatrix

    def normalized_confusion_matrix(self):
        if self.ConfusionMatrix is None:
            self.confusion_matrix()
        confusion_mat = tf.cast(self.ConfusionMatrix, tf.float32)
        denominator = tf.expand_dims(tf.reduce_sum(confusion_mat, axis=1), 1)
        self.NormalizedConfusionMatrix = confusion_mat / denominator
        return self.NormalizedConfusionMatrix

    def calculate_metrics(self):
        if self.ConfusionMatrix is None:
            self.ConfusionMatrix = self.confusion_matrix()

        num_classes = self.num_classes
        dtype = self.ConfusionMatrix.dtype
        self.TP = tf.diag_part(self.ConfusionMatrix)
        self.TN = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.FP = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.FN = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.Accuracy = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.Sensitivity = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.Specificity = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.Precision = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)
        self.F1Score = tf.Variable(tf.zeros([num_classes, ], dtype=dtype), dtype=dtype)

        sum_0 = tf.reduce_sum(self.ConfusionMatrix, axis=0)
        sum_1 = tf.reduce_sum(self.ConfusionMatrix, axis=1)
        sum_all = tf.reduce_sum(self.ConfusionMatrix)

        for i in range(num_classes):
            tn = sum_all - sum_0[i] - sum_1[i] + self.TP[i]
            fp = sum_0[i] - self.TP[i]
            fn = sum_1[i] - self.TP[i]

            self.TN = self.TN + tf.sparse_to_dense(sparse_indices=i, output_shape=[num_classes, ],
                                                   sparse_values=tn)
            self.FP = self.FP + tf.sparse_to_dense(sparse_indices=i, output_shape=[num_classes, ],
                                                   sparse_values=fp)
            self.FN = self.FN + tf.sparse_to_dense(sparse_indices=i, output_shape=[num_classes, ],
                                                   sparse_values=fn)

    def accuracy(self):
        if self.TP is None:
            self.calculate_metrics()
        self.Accuracy = tf.div(tf.cast(tf.reduce_sum(self.TP), tf.float32),
                               tf.cast(tf.reduce_sum(self.ConfusionMatrix), tf.float32))
        return self.Accuracy

    def accuracy_per_class(self):
        if self.TP is None:
            self.calculate_metrics()
        self.AccuracyClass = tf.div(tf.cast(self.TP, tf.float32),
                                    tf.cast(tf.reduce_sum(self.ConfusionMatrix, axis=1), tf.float32))
        return self.AccuracyClass

    def precision(self):
        if self.TP is None:
            self.calculate_metrics()

        self.Precision = tf.reduce_sum(self.TP)/tf.reduce_sum(self.TP + self.FP)
        return self.Precision

    def precision_per_class(self):
        if self.TP is None:
            self.calculate_metrics()

        self.PrecisionClass = self.TP/(self.TP + self.FP)
        return self.PrecisionClass

    def sensitivity(self):
        if self.TP is None:
            self.calculate_metrics()

        self.Sensitivity = tf.reduce_sum(self.TP) / tf.reduce_sum(self.TP + self.FN)
        return self.Sensitivity

    def sensitivity_per_class(self):
        if self.TP is None:
            self.calculate_metrics()

        self.SensitivityClass = self.TP / (self.TP + self.FN)
        return self.SensitivityClass

    def specificity(self):
        if self.TN is None:
            self.calculate_metrics()

        self.Specificity = tf.reduce_sum(self.TN) / tf.reduce_sum(self.FP + self.TN)
        return self.Specificity

    def specificity_per_class(self):
        if self.TN is None:
            self.calculate_metrics()

        self.SpecificityClass = self.TN / (self.FP + self.TN)
        return self.SpecificityClass

    def f1score(self):
        if self.TP is None:
            self.calculate_metrics()

        numerator = 2 * tf.cast(self.TP, tf.int32)
        denominator = numerator + self.FP + self.FN
        self.F1Score = tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
        return self.F1Score

    def f1score_per_class(self):
        if self.TP is None:
            self.calculate_metrics()

        numerator = 2 * tf.cast(self.TP, tf.int32)
        denominator = numerator + self.FP + self.FN
        self.F1ScoreClass = numerator / denominator
        return self.F1ScoreClass

    def variable_summaries(self):
        name = 'metrics'
        with tf.name_scope('MetricSummaries'):
            for i in range(self.num_classes):
                # tf.summary.scalar(name + '_TP_' + str(i), self.TP[i])
                # tf.summary.scalar(name + '_TN_' + str(i), self.TN[i])
                # tf.summary.scalar(name + '_FP_' + str(i), self.FP[i])
                # tf.summary.scalar(name + '_FN_' + str(i), self.FN[i])
                tf.summary.scalar(name + '_Accuracy_' + str(i), self.AccuracyClass[i])
                tf.summary.scalar(name + '_Precision_' + str(i), self.PrecisionClass[i])
                tf.summary.scalar(name + '_Sensitivity_' + str(i), self.SensitivityClass[i])
                tf.summary.scalar(name + '_Specificity_' + str(i), self.SpecificityClass[i])
                tf.summary.scalar(name + '_F1Score_' + str(i), self.F1ScoreClass[i])

        # tf.summary.scalar(name + '_TP', tf.reduce_sum(self.TP))
        # tf.summary.scalar(name + '_TN', tf.reduce_sum(self.TN))
        # tf.summary.scalar(name + '_FP', tf.reduce_sum(self.FP))
        # tf.summary.scalar(name + '_FN', tf.reduce_sum(self.FN))
        tf.summary.scalar(name + '_Accuracy', self.Accuracy)
        tf.summary.scalar(name + '_Precision', self.Precision)
        tf.summary.scalar(name + '_Sensitivity', self.Sensitivity)
        tf.summary.scalar(name + '_Specificity', self.Specificity)
        tf.summary.scalar(name + '_F1Score', self.F1Score)
        confusion_image = self.NormalizedConfusionMatrix
        confusion_image = tf.expand_dims(confusion_image, axis=0)
        confusion_image = tf.expand_dims(confusion_image, axis=3)
        tf.summary.image(name + '_confusion_matrix', confusion_image)

import tensorflow as tf
import numpy as np
import json


class VQAD4Model:
   def __init__(self,d4_params,data_params,token_to_idx, wordvec_dim=300,
                rnn_dim=256, rnn_num_layers=2, rnn_dropout=0):

    self.token_to_idx = token_to_idx
    self.batch_size = d4_params.batch_size
    self.max_length = data_params.max_length
    self.max_args = data_params.max_args

    def _add_placeholders(self):
        with tf.name_scope("input"):
            # [batch_size x max_length]
            self._pl_text = tf.placeholder(dtype=tf.int32,
                                           shape=[self.batch_size, self.max_length],
                                           name="text")
            # [batch_size]
            self._pl_text_len = tf.placeholder(dtype=tf.int64,
                                               shape=[self.batch_size],
                                               name="text_len")
            # [batch_size x max_args]
            self._pl_num_pos = tf.placeholder(dtype=tf.int32,
                                              shape=[self.batch_size, self.max_args],
                                              name="num_pos")
            # [batch_size x max_args]
            self._pl_nums = tf.placeholder(dtype=tf.float32,
                                           shape=[self.batch_size, self.max_args],
                                           name="nums")
            self.cnn_input = tf.placeholder(tf.float32)

    def cnn_lstm_model(self):
       with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_normal([len(token_to_idx), wordvec_dim],
                                                      mean=0.0, stddev=0.1, dtype=tf.float32),
                                     name="embedding_matrix")

            # [batch_size x max_length x vector_size]
            input_embedded = tf.nn.embedding_lookup(embeddings,
                                                    self._pl_text,
                                                    name="embedding_lookup")

       inputs = input_embedded
       with tf.name_scope("LSTM"):
           lstm_cell = tf.contrib.rnn.BasicLSTMCell(wordvec_dim,
                                                    forget_bias=1.0)
           outputs,_ = tf.nn.dynamic_rnn(lstm_cell, inputs,
                                               sequence_length=self._pl_text_len,
                                               dtype=tf.float32)

            # N, T = _pl_text.size()
            # idx = torch.LongTensor(N).fill_(T - 1)
            #
            # arr = np.asarray(all_questions, dtype=np.int64)
            # x = tf.constant(arr)
            # x = tf.cast(x, tf.int32)
            N, T = self._pl_text.get_shape()

            # if the object indexing dosent work, use the   arr = np.asarray(all_questions, dtype=np.int64) insted of _pl_text
            NULL = 0
            idx = np.array([], dtype='i')
            for i in range(N):
                for t in range(T - 1):
                    if self._pl_text[i, t] != NULL and self._pl_text[i, t + 1] == NULL:
                        idx = np.append(idx, [i, t])
                    else:
                        idx = np.append(idx, [i, 40])
                        break

            idx = idx.reshape(N, 2)
            idx = tf.convert_to_tensor(idx)

            # hs = tf.random_normal([7998, 41, 256])
            # hs = outputs
            output_rnn = tf.gather_nd(outputs, idx)

       with tf.name_scope("CNN"):
           filter = (1, 1, 1024, 512)
           N, _ = self._pl_text.get_shape()

           initial = tf.Variable(tf.truncated_normal(filter, stddev=0.1))  # W
           # x = tf.placeholder(tf.float32)  # input

           conv = tf.nn.conv2d(self.cnn_input, initial, strides=[1, 1, 1, 1], padding='SAME')
           relu = tf.nn.relu(conv)
           pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME')  # ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1]
           output_cnn = tf.reshape(pool, [N, -1])

       with tf.name_scope("MLP"):

            with open("/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/vocab.json", 'r') as f:  #should be in args
                vocab = json.load(f)['answer_token_to_idx']

            output_dim = len(vocab)
            a_matmul = tf.Variable(tf.random_normal([25344, 1024]))            #change for variable
            a_matmul2 = tf.Variable(tf.random_normal([1024, output_dim]))      #change for variable
            # cnn = tf.random_normal([1, 25088])                     # = to output_cnn
            # rnn = tf.random_normal([1, 256])                       # = to output_rnn
                                                                     #for loop (not sure)
            cat_feats = tf.concat(1, [output_rnn, output_cnn])
            cat_feats = tf.reshape(cat_feats, [-1])

            mean_x, std_x = tf.nn.moments(cat_feats, axes =[0])
            norm_1 = tf.nn.batch_normalization(cat_feats, mean_x, std_x, None, None, 1e-5)   #shape=(25344,)
            linear_1 = tf.matmul([norm_1], a_matmul)     #   shape=(25344,) ,   shape=(25344, 1024)
            linear_1 = tf.reshape(linear_1, [-1])

            mean_x_2, std_x_2 = tf.nn.moments(linear_1, axes =[0])
            norm_2 = tf.nn.batch_normalization(linear_1, mean_x_2, std_x_2, None, None, 1e-5)
            relu = tf.nn.relu(norm_2)     #shape=(1024,)
            linear_2 = tf.matmul([relu], a_matmul2)       #   shape=(1024,)  ,   shape=(1024, 6)

            output_mlp = linear_2



# if __name__ == "__main__":
#     # dataset load
#     data, MAX_LENGTH, MAX_ARGS, VOCAB_SIZE = load_all_datasets()
#
#     # parameter setup
#     MAX_LENGTH = 41
#     BATCH_SIZE = 5
#     VALUE_SIZE = 20
#     MIN_RETURN_WIDTH = VALUE_SIZE
#     STACK_SIZE = 10
#
#     d4_params = d4InitParams(stack_size=STACK_SIZE,
#                              value_size=VALUE_SIZE,
#                              batch_size=BATCH_SIZE,
#                              min_return_width=MIN_RETURN_WIDTH)
#
#     data_params = DataParams(vocab_size=VOCAB_SIZE,
#                              max_length=MAX_LENGTH,
#                              max_args=MAX_ARGS)
#
#     train_params = TrainParams(train=True,
#                                learning_rate=0.01,
#                                num_steps_train=6,
#                                num_steps_test=10)
#
#     def load_sketch_from_file(filename):
#         with open(filename, "r") as f:
#             scaffold = f.read()
#         return scaffold
#
#     sketch = load_sketch_from_file("./experiments/wap/algebra_sketch.d4")
#
#     print(sketch)
#
#     model = AlgebraD4Model(sketch, d4_params, data_params, train_params)
#
#     model.build_graph()
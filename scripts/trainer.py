import numpy as np
import tensorflow as tf
from data import ClevrDataset, ClevrDataLoader
from model import d4InitParams, DataParams, TrainParams, VQAD4Model
import utils
# np.set_printoptions(linewidth=20000, precision=5, suppress=True, threshold=np.nan)
np.set_printoptions(threshold=np.inf)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("test", -1, "secret stuff!")


def main(argv):

    # data, MAX_LENGTH, MAX_ARGS, VOCAB_SIZE = load_all_datasets()

    TOKEN_TO_IDX = utils.load_vocab("/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/vocab.json")['question_token_to_idx']

    # parameter setup
    MAX_LENGTH = 41
    BATCH_SIZE = 5
    VALUE_SIZE = 25088
    MIN_RETURN_WIDTH = VALUE_SIZE           #<<<
    STACK_SIZE = 10                         #<<<
    WORDVEC_DIM = 300
    RNN_DIM = 256
    IMG_SIZE= 14
    NUM_CHANNELS = 1024

    d4_params = d4InitParams(stack_size=STACK_SIZE,
                             value_size=VALUE_SIZE,
                             batch_size=BATCH_SIZE,
                             min_return_width=MIN_RETURN_WIDTH)

    data_params = DataParams(max_length=MAX_LENGTH,
                             wordvec_dim=WORDVEC_DIM,
                             rnn_dim=RNN_DIM,
                             img_size=IMG_SIZE,
                             num_channels=NUM_CHANNELS,
                             token_to_idx=TOKEN_TO_IDX)

    train_params = TrainParams(train=True,
                               learning_rate=0.02,
                               num_steps_train=7,
                               num_steps_test=7)


    def load_sketch_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold

    # building batcher
    # question_h5 = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/train_questions.h5'
    # feature_h5 = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/train_features.h5'
    # vocab = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/vocab.json'

    question_h5 = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/Dataset/onebatch_questions.h5'
    feature_h5 = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/Dataset/onebatch_features.h5'
    vocab = '/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/data/vocab.json'

    train_loader_kwargs = {
        'question_h5': question_h5,
        'feature_h5': feature_h5,
        'vocab': vocab,
        'batch_size': 1,
        'shuffle': 0,
        'question_families': None,
        'max_samples': None,
        'num_workers': 1,
    }
    batcher_train = ClevrDataLoader(**train_loader_kwargs)

    # build the model
    sketch = load_sketch_from_file("/Users/sma/Documents/PERSONALIZATION/NADAHARVARD/code/vqa/sketch_vqa.d4")
    # print(sketch)
    model = VQAD4Model(sketch, d4_params, data_params, train_params)
    model.build_graph()

    directory_save = "./tmp/wap/checkpoints/"
    import os
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)

    epoch_count = tf.Variable(0, name="epoch", trainable=False)
    max_accuracy = float('-inf')

    with tf.Session() as sess:

        # if True:
        #     model.load_model(sess, directory_save)
        #     print('loaded model')
        #     accuracy = model.run_eval_step(sess, data.dev, debug=False)
        #     print(accuracy)
        #     # accuracy, partial_accuracy = model.run_eval_step(sess, dataset_dev)
        #     exit(0)

        summary_writer = tf.train.SummaryWriter("./tmp/summaries/wap", tf.get_default_graph())
        sess.run(tf.initialize_all_variables())

        for epoch in range(50):
            epoch_count.assign_add(1)
            print('epoch', epoch)

            # TRAIN
            total_loss = 0.0
            for batch_no in range(batcher_train.batch_number):
                batch = batcher_train.next_batch()
                _, summaries, loss, global_step = model.run_train_step(sess, batch)
                # print('    Loss per batch: ', loss / BATCH_SIZE)
                total_loss += loss
                summary_writer.add_summary(summaries, global_step)

            total_loss /= (batcher_train.batch_number * BATCH_SIZE)
            print('  loss per epoch: ', total_loss)

            # logging summaries
            summary_loss = tf.Summary(
                value=[tf.Summary.Value(tag="loss_per_epoch", simple_value=total_loss)]
            )
            summary_writer.add_summary(summary_loss, epoch)
            summary_writer.flush()

            # if epoch % 10 == 0:
            #     model.run_test_step(sess, batch)

            print("  train set eval...")
            if epoch % 1 == 0:
                model.run_eval_step(sess, data.train, debug=False)

            print("  dev set eval...")
            if epoch % 1 == 0:
                accuracy = model.run_eval_step(sess, data.dev)
                if accuracy > max_accuracy and accuracy > 0.94:
                    max_accuracy = accuracy
                    print('Saving model...')

                    model.save_model(sess, directory_save + "model.checkpoint",
                                     global_step=global_step)
                    print('Model saved...')

                    # print("train set...")

                # for i_batch in range(BATCH_SIZE):
                #     model._dsm_loss.load_target_stack(batch.targets[i_batch], batch=i_batch)
                #
                #
                #
                # # model.run_eval_step(sess, batch)
                #
                # print('instance', batch.targets)
                # print('loss', loss)
                #
                # summaries, loss, global_step = model.run_eval_step(sess, batch)
                #
                # for i in range(0,12):
                #     if i == 0:
                #         t = tf.get_default_graph().get_tensor_by_name(
                #             "train_step/execute/choice_decoder/weights_22:0")
                #     else:
                #         t = tf.get_default_graph().get_tensor_by_name(
                #             "train_step_{0}/execute/choice_decoder/weights_22:0".format(i))
                #
                #     feed = model._build_birnn_feed(batch)
                #     feed = model._dsm_loss.current_feed_dict(feed)
                #     x = sess.run(t, feed_dict=feed)
                #     print(x)
                #     print('-')

        #
        # weights_22
        #     # fetch the slot variables to regularise them
        #     with tf.name_scope("slot_variables"):
        #         self.slot_variables = [tf.reshape(v, [-1])
        #                                for v in tf.trainable_variables()
        #                                if v.name.startswith("slots")]


if __name__ == "__main__":
    tf.app.run()

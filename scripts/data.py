import numpy as np
import h5py
import torch
import utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
torch.backends.cudnn.enabled = True
from torch.autograd import Variable


vocab = './vqa/data/vocab.json'

vocab = utils.load_vocab(vocab)
def _dataset_to_tensor(dset, mask=None):
  arr = np.asarray(dset, dtype=np.int64)
  if mask is not None:
    arr = arr[mask]
  tensor = torch.LongTensor(arr)
  return tensor

class Dataset_eval:
    def __init__(self, ids, seq_length, number_positions, numbers, target_values):
      self.tokens = ids
      self.seq_lengths = seq_length
      self.num_pos = number_positions
      self.nums = numbers
      self.targets = target_values


class ClevrDataset(Dataset):
  def __init__(self, question_h5, feature_h5, vocab, mode='prefix', max_samples=None, question_families=None,
               image_idx_start_from=None):

    # question_h5 = h5py.File(question_h5, 'r')
    mode_choices = ['prefix', 'postfix']
    if mode not in mode_choices:
      raise ValueError('Invalid mode "%s"' % mode)
    self.vocab = vocab
    self.feature_h5 = feature_h5
    self.mode = mode
    self.max_samples = max_samples

    mask = None
    if question_families is not None:
      # Use only the specified families
      all_families = np.asarray(question_h5['question_families'])
      N = all_families.shape[0]
      print(question_families)
      target_families = np.asarray(question_families)[:, None]
      mask = (all_families == target_families).any(axis=0)
    if image_idx_start_from is not None:
      all_image_idxs = np.asarray(question_h5['image_idxs'])
      mask = all_image_idxs >= image_idx_start_from

    # Data from the question file is small, so read it all into memory
    # print('Reading question data into memory')
    self.all_questions = _dataset_to_tensor(question_h5['questions'][:], mask)
    self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask)
    self.all_answers = _dataset_to_tensor(question_h5['answers'], mask)

  def __getitem__(self, index):
    question = self.all_questions[index]
    image_idx = self.all_image_idxs[index]
    answer = self.all_answers[index]
    feats = self.feature_h5['features'][image_idx]
    feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))


    return (question, feats, answer)

  def __len__(self):
    if self.max_samples is None:
      return self.all_questions.size(0)
    else:
      return min(self.max_samples, self.all_questions.size(0))

class ClevrDataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'question_h5' not in kwargs:
      raise ValueError('Must give question_h5')
    if 'feature_h5' not in kwargs:
      raise ValueError('Must give feature_h5')
    if 'vocab' not in kwargs:
      raise ValueError('Must give vocab')

    feature_h5_path = kwargs.pop('feature_h5')
    # print('Reading features from ', feature_h5_path)
    self.feature_h5 = h5py.File(feature_h5_path, 'r')

    vocab = kwargs.pop('vocab')
    mode = kwargs.pop('mode', 'prefix')

    question_families = kwargs.pop('question_families', None)
    max_samples = kwargs.pop('max_samples', None)
    question_h5_path = kwargs.pop('question_h5')
    image_idx_start_from = kwargs.pop('image_idx_start_from', None)
    # print('Reading questions from ', question_h5_path)
    with h5py.File(question_h5_path, 'r') as question_h5:
      self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                  max_samples=max_samples,
                                  question_families=question_families,
                                  image_idx_start_from=image_idx_start_from)
    kwargs['collate_fn'] = clevr_collate
    super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

  def close(self):
    if self.feature_h5 is not None:
      self.feature_h5.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  question_batch = default_collate(transposed[0])
  feat_batch = transposed[1]
  if any(f is not None for f in feat_batch):
    feat_batch = default_collate(feat_batch)
  answer_batch = default_collate(transposed[2])
  return [question_batch, feat_batch, answer_batch]

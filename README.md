# VQA using ∂4


## Installation

### Dependencies

Python 3
Pytorch 1.10.0

with

- Tensorflow 0.11.0

```
pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp35-cp35m-linux_x86_64.whl
```

## Use

### Dataset

Generate dataset form [here](https://github.com/facebookresearch/clevr-dataset-gen)
And use this [template](https://github.com/SamaherA/vqa/blob/main/dataset/compare_integer.json) for generating questions

##Note
We edit d4: [extensible_dsm.py](https://github.com/uclnlp/d4/blob/master/d4/dsm/extensible_dsm.py), line 275. We changed the type into float32:  create_alg_op_matrixret = np.zeros([size, size,size], dtype=np.float32)

### Running the experiment

#### Extract features

```
python3 vqa/extract_features.py \
--input_image_dir /data/images\
--output_h5_file data/train_features.h5

```

#### Process questions
Use this [vocal.json](https://github.com/SamaherA/vqa/blob/main/dataset/vocab.json) for vocabs

```
python3 vqa/preprocess_questions.py  \
--input_questions_json data/CLEVR_questions.json \
--input_vocab_json data/vocab.json \
--output_h5_file data/train_questions.h5

```


#### Training 
Use this [vocal.json](https://github.com/SamaherA/vqa/blob/main/dataset/vocab.json) for vocabs

```
python3  trainer.py
```



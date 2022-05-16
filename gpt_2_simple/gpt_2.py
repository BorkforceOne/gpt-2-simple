import os
import json
import requests
import shutil
import re
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
import time
import csv
from sys import exit

from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer

tf.compat.v1.disable_eager_execution()

def download_file_with_progress(url_base, sub_dir, model_name, file_name):
    """General utility for incrementally downloading files from the internet
    with progress bar
    from url_base / sub_dir / filename
    to local file system sub_dir / filename

    Parameters
    ----------
    file_name : str
        name of file to get e.g. "hparams.json"
    sub_dir: str
        subdirectory inside which to get and copy locally eg. "models/124M"
        no trailing slash
    url_base : str
        Start of URL location specifying server and any base directories no
        trailing slash
        e.g. "https://storage.googleapis.com/gpt-2"
    """

    # set to download 1MB at a time. This could be much larger with no issue
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    r = requests.get(url_base + "/models/" + model_name + "/" + file_name, stream=True)
    with open(os.path.join(sub_dir, file_name), 'wb') as f:
        file_size = int(r.headers["content-length"])
        with tqdm(ncols=100, desc="Fetching " + file_name,
                  total=file_size, unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)


def download_gpt2(model_dir='models', model_name='124M'):
    """Downloads the GPT-2 model into the current directory
    from Google Cloud Storage.

    Parameters
    ----------
    model_dir : str
        parent directory of model to download

    model_name : str
        name of the GPT-2 model to download.
        As of 22 May 2019 one of "124M" or "355M" but may later include other
        model sizes

    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    """

    # create the <model_dir>/<model_name> subdirectory if not present
    sub_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_dir = sub_dir.replace('\\', '/')  # needed for Windows

    for file_name in ['checkpoint', 'encoder.json', 'hparams.json',
                      'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                      'model.ckpt.meta', 'vocab.bpe']:
        download_file_with_progress(url_base="https://openaipublic.blob.core.windows.net/gpt-2",
                                    sub_dir=sub_dir,
                                    model_name=model_name,
                                    file_name=file_name)


def start_tf_sess(threads=-1, server=None):
    """
    Returns a tf.Session w/ config
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    if server is not None:
        return tf.compat.v1.Session(target=server.target, config=config)

    return tf.compat.v1.Session(config=config)


def reset_session(sess, threads=-1, server=None):
    """Resets the current TensorFlow session, to clear memory
    or load another model.
    """

    tf.compat.v1.reset_default_graph()
    sess.close()
    sess = start_tf_sess(threads, server)
    return sess

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class Finetune:
    def __init__(self,
             sess,
             dataset,
             model_name='124M',
             model_dir='models',
             combine=50000,
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             restore_from='latest',
             run_name='run1',
             checkpoint_dir='checkpoint',
             multi_gpu=False,
             print_every=1,
             max_checkpoints=1,
             use_memory_saving_gradients=False,
             only_train_transformer_layers=False,
             optimizer='adam',
             overwrite=False,
             target_loss=None,
             sample_size=1024,
             reuse=False):
        self.sess = sess
        self.dataset = dataset
        self.model_name = model_name
        self.model_dir = model_dir
        self.combine = combine
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accumulate_gradients = accumulate_gradients
        self.restore_from = restore_from
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.multi_gpu = multi_gpu
        self.print_every = print_every
        self.max_checkpoints = max_checkpoints
        self.use_memory_saving_gradients = use_memory_saving_gradients
        self.only_train_transformer_layers = only_train_transformer_layers
        self.optimizer = optimizer
        self.overwrite = overwrite
        self.target_loss = target_loss
        self.sample_size = sample_size
        self.reuse = reuse
        self.avg_loss = (0.0, 0.0)

    def maketree_(self, path):
        try:
            os.makedirs(path)
        except:
            pass

    def init(self):
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.run_name)

        self.maketree_(self.checkpoint_path)
        self.files = [f for f in os.listdir(self.checkpoint_path)]
        for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
            try:
                shutil.copyfile(os.path.join(self.model_dir, self.model_name, file),
                                os.path.join(self.checkpoint_path, file))
            except FileNotFoundError as fnf_error:
                print("You need to download the GPT-2 model first via download_gpt2()")
                raise(fnf_error)

        enc = encoder.get_encoder(self.checkpoint_path)
        hparams = model.default_hparams()
        with open(os.path.join(self.checkpoint_path, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if self.model_name not in ['117M', '124M']:
            print('For larger models, the recommended finetune() parameters are:')
            print('\tuse_memory_saving_gradients = True')
            print('\tonly_train_transformer_layers = True')
            print('\taccumulate_gradients = 1\n')

        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        gpus = []

        if self.multi_gpu:
            gpus = get_available_gpus()

        self.output = model.model(hparams=hparams, X=self.context, gpus=gpus, reuse=self.reuse)
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.context[:, 1:], logits=self.output['logits'][:, :-1]))

        all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if self.only_train_transformer_layers else all_vars

        if self.optimizer == 'adam':
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        if tf.__version__ >= '2.0.0' and self.use_memory_saving_gradients:
            exit("Memory saving gradients are not implemented for Tensorflow 2 yet.")

        if self.accumulate_gradients > 1:
            if self.use_memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            self.opt_reset = opt.reset()
            self.opt_compute = opt.compute_gradients(self.loss)
            self.opt_apply = opt.apply_gradients()
            self.summary_loss = tf.compat.v1.summary.scalar('loss', self.opt_apply)
        else:
            if self.use_memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(self.loss, train_vars)
            else:
                opt_grads = tf.gradients(ys=self.loss, xs=train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            self.opt_apply = opt.apply_gradients(opt_grads)
            self.summary_loss = tf.compat.v1.summary.scalar('loss', self.loss)

        self.summary_log = tf.compat.v1.summary.FileWriter(self.checkpoint_path)

        self.saver = tf.compat.v1.train.Saver(
            var_list=all_vars,
            max_to_keep=self.max_checkpoints)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(self.checkpoint_path)
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join(self.model_dir, self.model_name))
        elif self.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(self.model_dir, self.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(self.restore_from)
        print('Loading checkpoint', ckpt)
        self.saver.restore(self.sess, ckpt)

        print('Loading dataset...')
        chunks = load_dataset(enc, self.dataset, self.combine)
        self.data_sampler = Sampler(chunks)
        print('dataset has', self.data_sampler.total_size, 'tokens')

        # Load counter info
        self.counter = 1
        self.counter_path = os.path.join(self.checkpoint_path, 'counter')
        if os.path.exists(self.counter_path) and self.restore_from == 'latest':
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(self.counter_path, 'r') as fp:
                self.counter = int(fp.read()) + 1
        
        self.counter_base = self.counter
        self.start_time = time.time()

    def save(self):
        self.maketree_(self.checkpoint_path)
        print(
            'Saving',
            os.path.join(self.checkpoint_path,
                        'model-{}').format(self.counter-1))
        self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_path, 'model'),
            global_step=self.counter-1)
        with open(self.counter_path, 'w') as fp:
            fp.write(str(self.counter-1) + '\n')

    def train(self):
        def sample_batch():
            return [self.data_sampler.sample(self.sample_size) for _ in range(self.batch_size)]

        if self.overwrite and self.restore_from == 'latest':
            for file in self.files:
                if file.startswith('model') or file.startswith('events'):
                    os.remove(os.path.join(self.checkpoint_path, file))
            self.save()

        if self.accumulate_gradients > 1:
            self.sess.run(self.opt_reset)
            for _ in range(self.accumulate_gradients):
                self.sess.run(
                    self.opt_compute, feed_dict={self.context: sample_batch()})
            (v_loss, v_summary) = self.sess.run((self.opt_apply, self.summary_loss))
        else:
            (_, v_loss, v_summary) = self.sess.run(
                (self.opt_apply, self.loss, self.summary_loss),
                feed_dict={self.context: sample_batch()})

        self.summary_log.add_summary(v_summary, self.counter)

        if self.counter % self.print_every == 0:
            self.avg_loss = (self.avg_loss[0] * 0.99 + v_loss,
                        self.avg_loss[1] * 0.99 + 1.0)

            print(
                '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                .format(
                    counter=self.counter,
                    time=time.time() - self.start_time,
                    loss=v_loss,
                    avg=self.avg_loss[0] / self.avg_loss[1]))

        self.counter += 1


def load_gpt2(sess,
              checkpoint='latest',
              run_name="run1",
              checkpoint_dir="checkpoint",
              model_name=None,
              model_dir='models',
              multi_gpu=False,
              reuse=False):
    """Loads the model checkpoint or existing model into a TensorFlow session
    for repeated predictions.
    """

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])

    gpus = []
    if multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus, reuse=reuse)

    if checkpoint=='latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
    else:
        ckpt = os.path.join(checkpoint_path,checkpoint)

    saver = tf.compat.v1.train.Saver(allow_empty=True)
    sess.run(tf.compat.v1.global_variables_initializer())

    if model_name:
        print('Loading pretrained model', ckpt)
    else:
        print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)


def generate(sess,
             run_name='run1',
             checkpoint_dir='checkpoint',
             model_name=None,
             model_dir='models',
             sample_dir='samples',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             top_p=0.0,
             include_prefix=True):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if prefix:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        context_tokens = enc.encode(prefix)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams,
        length=min(length, 1023 - (len(context_tokens) if prefix else 0)),
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )[:, 1:]

    if destination_path:
        f = open(destination_path, 'w')
    generated = 0
    gen_texts = []
    while generated < nsamples:
        if not prefix:
            out = sess.run(output)
        else:
            out = sess.run(output, feed_dict={
                    context: batch_size * [context_tokens]
                })
        for i in range(batch_size):
            generated += 1
            gen_text = enc.decode(out[i])
            if prefix:
                gen_text = enc.decode(context_tokens[:1]) + gen_text
            if truncate:
                truncate_esc = re.escape(truncate)
                if prefix and not include_prefix:
                    prefix_esc = re.escape(prefix)
                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                         truncate_esc)
                else:
                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                trunc_text = re.search(pattern, gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            gen_text = gen_text.lstrip('\n')
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim), end='')
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts


def generate_to_file(sess,
                     run_name='run1',
                     checkpoint_dir='checkpoint',
                     model_name=None,
                     model_dir='models',
                     truncate=None,
                     destination_path='gpt_2_gen_texts.txt',
                     sample_delim='=' * 20 + '\n',
                     prefix=None,
                     seed=None,
                     nsamples=1,
                     batch_size=1,
                     length=1023,
                     temperature=0.7,
                     top_k=0,
                     top_p=0.0,
                     include_prefix=True):
    """Generates the texts to a file.

    sample_delim separates texts: set to '' if each text is a small document.

    Adapted from https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn.py
    """

    generate(sess=sess,
             run_name=run_name,
             checkpoint_dir=checkpoint_dir,
             model_name=model_name,
             model_dir=model_dir,
             return_as_list=False,
             truncate=truncate,
             destination_path=destination_path,
             sample_delim=sample_delim,
             prefix=prefix,
             seed=seed,
             nsamples=nsamples,
             batch_size=batch_size,
             length=length,
             temperature=temperature,
             top_k=top_k,
             top_p=top_p,
             include_prefix=include_prefix)


def encode_csv(csv_path, out_path='csv_encoded.txt', header=True,
               start_token="<|startoftext|>",
               end_token="<|endoftext|>"):
    """Encodes a single-column CSV to a format suitable for gpt-2-simple.
       Automatically adds the specified prefix and suffix tokens.
    """

    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f:
        with open(out_path, 'w', encoding='utf8', errors='ignore') as w:
            if header:
                f.readline()
            reader = csv.reader(f)
            for row in reader:
                w.write(start_token + row[0] + end_token + "\n")


def encode_dataset(file_path, model_dir='models', out_path='text_encoded.npz',
                   model_name="124M",
                   combine=50000):
    """Preencodes a text document into chunks and compresses it,
    saving time when generated.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/encode.py
    """

    model_path = os.path.join(model_dir, model_name)
    enc = encoder.get_encoder(model_path)
    print('Reading files')
    chunks = load_dataset(enc, file_path, combine)
    print('Writing', out_path)
    np.savez_compressed(out_path, *chunks)


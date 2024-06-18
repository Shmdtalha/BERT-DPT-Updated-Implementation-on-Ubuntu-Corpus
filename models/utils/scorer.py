import numpy as np

def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
  total_num_split = len(ground_truth) / eval_candidates_num

  pred_split = np.split(prediction, total_num_split)
  gt_split = np.split(np.array(ground_truth), total_num_split)
  orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
  stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)

  rank_by_pred_l = []
  for i, stack_score in enumerate(stack_scores):
    rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
    rank_by_pred = np.stack(rank_by_pred, axis=-1)
    rank_by_pred_l.append(rank_by_pred[0])

  return np.array(rank_by_pred_l)

def logits_recall_at_k(rank_by_pred, k_list=[1, 2, 5, 10]):
  # 1 dialog, 10 response candidates ground truth 1 or 0
  # prediction_score : [batch_size]
  # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
  # e.g. batch : 100 -> 100/10 = 10

  num_correct = np.zeros([rank_by_pred.shape[0], len(k_list)])

  pos_index = []
  for sorted_score in rank_by_pred:
    for p_i, score in enumerate(sorted_score):
      if int(score) == 1:
        pos_index.append(p_i)
  index_dict = dict()
  for i, p_i in enumerate(pos_index):
    index_dict[i] = p_i

  for i, p_i in enumerate(pos_index):
    for j, k in enumerate(k_list):
      if p_i + 1 <= k:
        num_correct[i][j] += 1

  return np.sum(num_correct, axis=0), pos_index

def logits_mrr(rank_by_pred):
  pos_index = []
  for sorted_score in rank_by_pred:
    for p_i, score in enumerate(sorted_score):
      if int(score) == 1:
        pos_index.append(p_i)

  # print("pos_index", pos_index)
  mrr = []
  for i, p_i in enumerate(pos_index):
    mrr.append(1 / (p_i + 1))

  # print(mrr)

  return np.sum(mrr)

'''
    def rouge():
    import numpy as np
    import re
    from rouge import Rouge
    from nltk.tree import *
    from nltk.parse import CoreNLPParser
    from nltk.tokenize import sent_tokenize
    import tensorflow as tf
    import nltk
    import collections
    import math
    from glob import glob
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    
    def print_args(flags):
        """Print arguments."""
        print("\nParameters:")
        for attr in flags:
            value = flags[attr].value
            print("{}={}".format(attr, value))
        print("")
    
    
    def load_embedding(embed_file, vocab):
        emb_dict = dict()
        emb_size = tf.flags.FLAGS.embedding_dim
        with open(embed_file, 'r', encoding='utf8') as f:
            for line in f:
                tokens = line.strip().split(" ")
                word = tokens[0]
                vec = list(map(float, tokens[1:]))
                emb_dict[word] = vec
                if emb_size:
                    assert emb_size == len(vec), "All embedding size should be same."
                else:
                    emb_size = len(vec)
        oov_counter = 0
        for token in vocab:
            if token not in emb_dict:
                emb_dict[token] = [0.0] * emb_size
                oov_counter +=1
        print('oove:', oov_counter, 'total dic:', len(emb_dict))
        with tf.device('/cpu:0'), tf.compat.v1.name_scope("embedding"):
        #with tf.variable_scope("pretrain_embeddings", dtype=dtype):
            emb_table = np.array([emb_dict[token] for token in vocab], dtype=np.float32)
            emb_table = tf.convert_to_tensor(value=emb_table, dtype=tf.float32)
            print('---- emb_table:', emb_table)
    
        return emb_dict, emb_size, emb_table
    
    def clean_str(string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    
    
    def load_vocab(vocab_file):
        """load vocab from vocab file.
        Args:
            vocab_file: vocab file path
        Returns:
            vocab_table, vocab, vocab_size
        """
        
        vocab_table = tf.python.ops.lookup_ops.index_table_from_file( # Returns a lookup table that converts a string tensor into int64 IDs.
            vocabulary_file=vocab_file, default_value=0)
        vocab = []
        with open(vocab_file, "rb") as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
        return vocab_table, vocab, vocab_size
    
    
    
    def load_model(sess, ckpt):
        with sess.as_default(): 
            with sess.graph.as_default(): 
                init_ops = [tf.compat.v1.global_variables_initializer(),
                            tf.compat.v1.local_variables_initializer(), tf.compat.v1.tables_initializer()]
                sess.run(init_ops)
                ckpt_path = tf.train.latest_checkpoint(ckpt)
                print("Loading saved model: " + ckpt_path)
                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, ckpt_path)
    
    # The code for batch iterator:
    def _parse_infer_csv(line):
        cols_types = [['']] * 3
        columns = tf.io.decode_csv(records=line, record_defaults=cols_types, field_delim='\t')
        return columns
    
    def _parse_infer_test_csv(line):
        cols_types = [['']] * 2
        columns = tf.io.decode_csv(records=line, record_defaults=cols_types, field_delim='\t')
        return columns
    
    def _truncate(tensor):
        dim = tf.size(input=t)
        return tf.cond( pred=tf.greater(dim, k), true_fn=lambda: tf.slice(t, [0], [k]))
    
    def _split_string(tensor):
        results = np.array(re.split('\[|\]|, |,', tensor.decode("utf-8") ))
        results = [float(result) for result in results if result!='']
        return np.array(results).astype(np.float32)
    
    
    
    def get_iterator(data_file, vocab_table, batch_size, max_seq_len, padding=True,):
        """Iterator for train and eval.
        Args:
            data_file: data file, each line contains a sentence that must be ranked
            vocab_table: tf look-up table
            max_seq_len: sentence max length
            padding: Bool
                set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
        Returns:
            (batch, size)
        """
        # interleave is very important to process multiple files at the same time
        dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
        dataset = dataset.map(_parse_infer_csv)
        dataset = dataset.map(lambda score, sent, feats: (tf.strings.to_number(score, tf.float32), tf.compat.v1.string_split([sent]).values,\
                              tf.compat.v1.py_func(_split_string, inp=[feats], Tout=tf.float32)))
        #                        tf.string_split([feats], delimiter=',] ' ).values)) # you can set num_parallel_calls 
        dataset = dataset.map(lambda score, sent_tokens, feats: (score, tf.cond(pred=tf.greater(tf.size(input=sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                         true_fn=lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                         false_fn=lambda: sent_tokens), feats)) # truncate to max_seq_length
        # Convert the word strings to ids.  Word strings that are not in the
        # vocab get the lookup table's default_value integer.
        dataset = dataset.map(lambda score, sent_tokens, feats:{'scores':score, 'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
        if padding:
            batch_dataset = dataset.padded_batch(batch_size, padded_shapes={'scores':[],'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]},
                                            padding_values=None,
                                            drop_remainder=False)
        else:
            batch_dataset = dataset.padded_batch(batch_size,padded_shapes={'scores':[],'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]}, drop_remainder=False)
        batched_iter = tf.compat.v1.data.make_initializable_iterator(batch_dataset)
        next_batch = batched_iter.get_next()
    
        return batched_iter, next_batch
    
    '''
    '''
    def _pad_up_to(tensor):
        constant_values = 'None'
        s = tf.shape(tensor)
        paddings = [[0,tf.flags.FLAGS.max_seq_len - tensor.shape[0]]]  
        return tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)
    
    def get_dev_data(data_file, vocab_table, batch_size, max_seq_len, padding=True,):
        dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
        dataset = dataset.map(_parse_infer_csv)
        dataset = dataset.map(lambda score, sent, feats: (tf.string_to_number(score, tf.float32), tf.string_split([sent]).values,\
                              tf.py_func(_split_string, inp=[feats], Tout=tf.float32)))
        dataset = dataset.map(lambda score, sent_tokens, feats: (score, tf.cond(tf.greater(tf.size(sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                         lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                         lambda: sent_tokens), feats)) # truncate to max_seq_length
        dataset = dataset.map(lambda score, sent_tokens, feats:(score,tf.py_function(_pad_up_to, inp=[sent_tokens], Tout=tf.string),feats))
        dataset = dataset.map(lambda score, sent_tokens, feats:{'scores':score, 'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
        iter = dataset.make_initializable_iterator()
        next_batch = iter.get_next()
        return iter, next_batch
    '''
    
    '''
    def get_test_iterator(data_file, 
                     vocab_table,
                     batch_size,
                     max_seq_len,
                     padding=True,):
    
        # interleave is very important to process multiple files at the same time
        dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
        dataset = dataset.map(_parse_infer_test_csv)
        dataset = dataset.map(lambda  sent, feats: (tf.compat.v1.string_split([sent]).values,\
                              tf.compat.v1.py_func(_split_string, inp=[feats], Tout=tf.float32)))
        #                        tf.string_split([feats], delimiter=',] ' ).values)) # you can set num_parallel_calls 
        dataset = dataset.map(lambda sent_tokens, feats: ( tf.cond(pred=tf.greater(tf.size(input=sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                         true_fn=lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                         false_fn=lambda: sent_tokens), feats)) # truncate to max_seq_length
        # Convert the word strings to ids.  Word strings that are not in the
        # vocab get the lookup table's default_value integer.
        dataset = dataset.map(lambda sent_tokens, feats:{'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
        if padding:
            batch_dataset = dataset.padded_batch(batch_size, padded_shapes={'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]},
                                            padding_values=None,
                                            drop_remainder=False)
        else:
            batch_dataset = dataset.padded_batch(batch_size,padded_shapes={'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]}, drop_remainder=False)
        batched_iter = tf.compat.v1.data.make_initializable_iterator(batch_dataset)
        next_batch = batched_iter.get_next()
    
        return batched_iter, next_batch

'''

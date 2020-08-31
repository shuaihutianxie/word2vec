from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename) #gettempdir()则用于返回保存临时文件的文件夹路径,os.path.join()函数：连接两个或更多的路径名组件
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename) #将URL表示的网络对象复制到本地文件
  statinfo = os.stat(local_filename) #os.stat() 方法用于在给定的路径上执行一个系统 stat 的调用
  if statinfo.st_size == expected_bytes: # stat.st_size: 普通文件以字节为单位的大小；包含等待某些特殊文件的数据
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


filename = maybe_download('text8.zip', 31344016)

print (filename)
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  # Counter(a)   计算a中每个元素的数量，按数量从大到小输出；经过Counter()方法排列后，获取数量最多的元素及准确数量，如果most_common()的参数是2，则获取数量排在前两位的元素及具体数量
  count.extend(collections.Counter(words).most_common(n_words - 1)) #collections模块中的Counter方法可以对列表和字符串进行计数，most_common可以用来统计列表或字符串中最常出现的元素
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
# batch_size 一个batch的个数；
# num_skips上下文的个数；
# skip_window 窗口的大小
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin #可以指定队列的长度
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span]) #[5234, 3081, 12]
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window] # -->[0, 2]
    words_to_use = random.sample(context_words, num_skips) #random.sample的用法，多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序; --> [0, 2]
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.（词的维度128）
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.(负采样个数)

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.

'''
np.random.choice(a=5, size=3, replace=False, p=None) 参数意思分别是从a中以概率P,
随机选择3个 p没有指定的时候相当于是一致的分布，a为一维数组或int，如果是int，则生成随机样本，就好像a是np.arange(N)一样
'''
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph() #tf.Graph()表示实例化一个用于tensorflow计算和表示用的数据流图

'''
1、使用g = tf.Graph()函数创建新的计算图
2、在with g.as_default():语句下定义属于计算图g的张量和操作
3、在with tf.Session()中通过参数graph=xxx指定当前会话所运行的计算图
4、如果没有显示指定张量和操作所属的计算图，则这些张量和操作属于默认计算图
5、一个图可以在多个sess中运行，一个sess也能运行多个图
(1)
https://blog.csdn.net/TeFuirnever/article/details/88928871
tf.placeholder()函数作为一种占位符用于定义过程，可以理解为形参，在执行的时候再赋具体的值。

tf.placeholder(
    dtype,
    shape=None,
    name=None
    )
参数：
  dtype：数据类型。常用的是tf.float32，tf.float64等数值类型
  shape：数据形状。默认是None，就是一维值，也可以多维，比如：[None，3]，表示列是3，行不一定
  name：名称

返回：
Tensor类型
  此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值。不必指定初始值，可在运行时，
  通过 Session.run 的函数的 feed_dict 参数指定。这也是其命名的原因所在，仅仅作为一种占位符。
  
(2)tf.constant, 在TensorFlow API中创建常量的函数原型如下所示
 https://blog.csdn.net/csdn_jiayu/article/details/82155224

(3) tf.name_scope('inputs'), 用于定义Python op的上下文管理器
 
(7) tf.Variable（initializer， name）：initializer是初始化参数，可以有tf.random_normal，tf.constant，tf.constant等，name就是变量的名字
  tf.Variable是一个Variable类。通过变量维持图graph的状态，以便在sess.run()中执行；可以用Variable类创建一个实例在图中增加变量；
  https://blog.csdn.net/UESTC_C2_403/article/details/72328296

'''
with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'): #用于定义Python op的上下文管理器
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  '''
    (4)tf.device()指定模型运行的具体设备，可以指定运行在GPU还是CUP上，以及哪块GPU上, tensorflow中不同的GPU使用/gpu:0和/gpu:1区分，而CPU不区分设备号，统一使用 /cpu:0
    https://blog.csdn.net/dcrmg/article/details/79747882
    
    (5) tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的
    https://blog.csdn.net/weixin_38859557/article/details/80878229
    
    (6) tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
    https://blog.csdn.net/yangfengling1023/article/details/82910951
    
    (7) tf.truncated_normal()函数介绍和示例 tf.truncated_normal(shape, mean, stddev)
    释义：截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
    shape，生成张量的维度
    mean，均值
    stddev，标准差
  '''
  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  '''
  (9) tf.nn.nce_loss
    https://blog.csdn.net/qq_36092251/article/details/79684721
  '''
  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled, #采样出多少个负样本
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  '''
   1) 、这是一个实现实现梯度下降算法的优化器类，用于构造一个新的梯度下降优化器实例
    tf.train.GradientDescentOptimizer(learning_rate, use_locking=False,name=’GradientDescent’)
     参数：
        learning_rate: A Tensor or a floating point value. 要使用的学习率
        use_locking: 要是True的话，就对于更新操作（update operations.）使用锁
        name: 名字，可选，默认是”GradientDescent”
     方法：
        minimize() 函数处理了梯度计算和参数更新两个操作
        compute_gradients() 函数用于获取梯度
        apply_gradients() 用于更新参数
        
   2）、tf.matmul（）将矩阵a乘以矩阵b，生成a * b
      https://www.cnblogs.com/AlvinSui/p/8987707.html
  '''
  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True) #tf.matmul（）将矩阵a乘以矩阵b，生成a * b

  # Merge all summaries.
  merged = tf.summary.merge_all()

  # Add variable initializer.
  init = tf.global_variables_initializer()

  # Create a saver.
  saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph) #指定一个文件用来保存图,https://blog.csdn.net/haoshan4783/article/details/89970227

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    '''
      函数参数:
        run(fetches, feed_dict=None, options=None, run_metadata=None)
        tf.Session.run() 执行 fetches 中的操作，计算 fetches 中的张量值
        这个函数执行一步 TensorFlow 运算，通过运行必要的图块来执行每一个操作，并且计算每一个 fetches 中的张量的值，用相关的输入变量替换 feed_dict 中的值。
        https://blog.csdn.net/huahuazhu/article/details/76178681
    '''
    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val

    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1] #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

  # Write corresponding labels for the embeddings.
  with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')

  # Save the model for checkpoints.
  saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

  #####################使用TensorBoard显示图##########################
  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.

# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne.png')

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
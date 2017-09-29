# -*- coding: utf-8 -*-  
#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# write by liyao, reference to the word2vec_basic.py file.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import json

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#设置字向量应用参数
flags=tf.app.flags
flags.DEFINE_string("datapath", "./data", "Directory to Sample data")
flags.DEFINE_string("modelpath", "./model", "Director to Save model data")
flags.DEFINE_integer("vocabulary_size", 5000, "The top N words size.")
flags.DEFINE_integer("batch_size", 128, "Train batch size.")
flags.DEFINE_integer("embedding_size", 128, "The embedding dimension size.")
flags.DEFINE_integer("skip_window", 1, "How many words to consider left and right")
flags.DEFINE_integer("num_skips", 2, "How many times to reuse an input to generate a label.")
flags.DEFINE_integer("valid_size", 16, "Random set of words to evaluate similarity on.")
flags.DEFINE_integer("valid_window", 100, "Only pick dev samples in the head of the distribution.")
flags.DEFINE_integer("num_sampled", 64, "Number of negative examples to sample.")
flags.DEFINE_integer("use_gpu", True, "Use gpu or cpu, True is use gpu")
flags.DEFINE_float("learn_rate", 0.5, "Initial learning rate.")

FLAGS = flags.FLAGS

#缺省参数设置
print("Default args:")
print("python cword2vec.py")
print("--datapath=./data")
print("--modelpath=./model")
print("--vocabulary_size=5000")
print("--batch_size=128")
print("--embedding_size=128")
print("--skip_window=1")
print("--num_skips=2")
print("--valid_size=16")
print("--valid_window=100")
print("--num_sampled=64")
print("--use_gpu=True")
print("--learn_rate=0.5")

#Options对象用于封装所有的参数, 包括输入输出路径和建模参数
class Options(object):
    #self.dataPath='./data/text'
    #self.modelPath='./model'
    #self.vocabulary_size = 5000
    #self.batch_size=128
    #self.embedding_size=128
    #self.skip_window=1
    #self.num_skips=2
    #self.valid_size = 16     # Random set of words to evaluate similarity on.
    #self.valid_window = 100  # Only pick dev samples in the head of the distribution.
    #self.num_sampled = 64    # Number of negative examples to sample.
    #self.use_gpu = True
    #self.learn_rate = 0.5
    def __init__(self):
        self.dataPath=FLAGS.datapath
        self.modelPath=FLAGS.modelpath
        self.vocabulary_size = FLAGS.vocabulary_size
        self.batch_size = FLAGS.batch_size
        self.embedding_size = FLAGS.embedding_size
        self.skip_window = FLAGS.skip_window
        self.num_skips = FLAGS.num_skips
        self.valid_size = FLAGS.valid_size
        self.valid_window = FLAGS.valid_window
        self.num_sampled = FLAGS.num_sampled
        self.use_gpu = FLAGS.use_gpu
        self.learn_rate=FLAGS.learn_rate
#end of class Options

#DataSet对象用于封装输入的训练文件
#从训练文件中生成文字编码id, 并形成训练的数据结合
class DataSet(object):
    #self.filelist  训练的文件名列表
    #self.count  训练文件中所有出现的文字频度排序列表
    #self.dictionary  将文字转换成编码id的词典对象
    #self.reverse_dictionary  由编码id反向查询文字的词典对象
    #self.fileListIndex 训练过程中使用文件的指针索引
    #self.data  训练过程中的文字编码id列表缓存
    #self.data_index  编码id列表指针

    #对象初始化
    def __init__(self, dpath, mpath):
        self.get_file_list(dpath)
        self.datapath=dpath
        self.modelpath=mpath
        self.dictionary={}
        self.fileListIndex=0
        self.data=None
        self.data_index = 0

    # 检索输入数据目录data_path, 列出所有文件名装入filelist列表
    def get_file_list(self, dpath):
        self.filelist=[]
        nlist=os.listdir(dpath)
        for i in range(0,len(nlist)):
            filename=os.path.join(dpath,nlist[i])
            self.filelist.append(filename)

    # 从文件中读取每个文字, 形成列表(去除回车)
    def read_data(self, filename):
        wdata=[]
        with open(filename) as f:
            print('Read data from '+filename+'.')
            for line in f.readlines():
                for ch in line:
                    if ch != "\n": wdata.append(ch)
        return wdata

    # 将所有输入文件中的文字扫描后
    # 按照出现频度排序
    # 对前topn的文字按照频度序号编码
    # 存入词典对象
    def build_dict(self, n_words):
        #临时字典
        d={}
        for filename in self.filelist:
            wordlist=self.read_data(filename)
            #print('Data size',len(wordlist))
            # 对单个输入文件中的每个文字频次进行统计
            word_count=collections.Counter(wordlist)
            #将各个文件中的字数统计以字为key进行累加，存储在临时字典中
            for count_key in word_count.keys():
                if count_key in d:
                    d[count_key]=d[count_key]+word_count[count_key]
                else:
                    d[count_key]=word_count[count_key]
        self.count=[['UNK',-1]]
        #统计所有文件中各文字出现的频度
        self.count.extend(collections.Counter(d).most_common(n_words - 1))
        #构造词典，将字映射成为id，id越小代表越常用，相当于词袋模型
        for word, _ in self.count:
            self.dictionary[word]=len(self.dictionary)
        #统计非常用字的出现次数，计入UNK，更新unk_count
        allcount=collections.Counter(d)
        unk_count=0
        for w,c in allcount.items():
            if w not in self.dictionary:
                unk_count += c
        self.count[0][1]=unk_count
        #构造反向词典，用于从id查询字
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    #使用json保存词袋模型检索词典和反向检索词典
    def save_dict(self):
        jsondict=open(self.modelpath+'/dictionary.json', 'w', encoding='utf-8')
        json.dump(self.dictionary, jsondict, ensure_ascii=False)
        jsondict.close()
        jsonrevdict=open(self.modelpath+'/reverse_dictionary.json', 'w', encoding='utf-8')
        json.dump(self.reverse_dictionary, jsonrevdict, ensure_ascii=False)
        jsonrevdict.close()
        print('save dictionary.')

    #直接从文件中加载检索词典
    def load_dict(self):
        jsondict=open(self.modelpath+'/dictionary.json', 'r', encoding='utf-8')
        self.dictionary=json.load(jsondict)
        jsondict.close()
        jsonrevdict=open(self.modelpath+'/reverse_dictionary.json', 'r', encoding='utf-8')
        self.reverse_dictionary=json.load(jsonrevdict)
        jsonrevdict.close()
        print('load dictionary.')
        
    # 从输入文件中抽取训练的数据集
    # 读取文件中的每个文字, 将文字按照词袋模型转化成id
    # 将转化好的id按照文字出现的次序形成列表保存在self.data中
    # 去除回车符
    def get_next_dataset(self):
        if self.fileListIndex >= len(self.filelist):
            self.data=None
        else:
            filename=self.filelist[self.fileListIndex]
            print("generate dataset from file %s."%filename)
            self.fileListIndex += 1
            words=self.read_data(filename)
            self.data=[]
            for word in words:
                if word in self.dictionary:
                    index=self.dictionary[word] 
                else:
                    index=0
                self.data.append(index)
            #print(self.data)

    # 构建训练的批次
    # skip-gram模型的训练数据结构是一个中心词上下各定义了一个窗口长度的词串
    # target中心词 
    # skip_window是中心池的上下窗口
    # span是中心词加上左右窗口的长度, 相当于*****#*****的结构, *的长度5是skip_windows, #是中心词, span是skip_windows*2+1
    # num_skips是数据在同一批次中重复使用的次数,一般设置为2, 如果样本多,应该设置比较小的值
    def generate_batch(self, batch_size, num_skips, skip_window):
        #断言batch_size是num_skips是的整数倍, 后面循环中会使用到
        # num_skips小于等于两倍的skip_window
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        # skip-gram使用中心词预测上下文, batch和labels的维度都是1
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # [ skip_window target skip_window ]
        span = 2 * skip_window + 1  
        #buffer是一个两端队列, 最大长度是span=3
        buffer = collections.deque(maxlen=span)
        # 如果数据指针到尾部, 切换文件
        if self.data_index + span >= len(self.data):
            print('switch file.')
            self.get_next_dataset()
            if type(self.data)==type(None) :
                return None, None
            self.data_index = 0
        #从data中取出span长的切片
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        #按照批次的数量循环, 因为num_skips定义重用的次数, 因此循环数量实际是batch_size整除num_skips
        #labels的构造是对buffer中的元素随机无放回的抽取(除去target)
        for i in range(batch_size // num_skips):
            #target指针设置为buffer的中心位置, 用于排除这个位置
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            # 数据重用的循环
            for j in range(num_skips):
                # 如果labels存在于targets_to_avoid中则规避, 否则一直在buffer范围内随机抽取
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                # batch样本取中心词
                # labels随机设置为target指向的单词id
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
                buffer.append(self.data[self.data_index])
                self.data_index+=1
                if self.data_index == len(self.data):
                    # 如果指针到尾部, 则切换文件
                    print('switch file.')
                    self.get_next_dataset()
                    if type(self.data)==type(None) :
                        return None, None
                    self.data_index=0
        # Backtrack a little bit to avoid skipping words in the end of a batch
        # 由于循环前先加了span,所以回退span
        self.data_index = self.data_index - span
        if self.data_index<0:
            self.data_index=0
        return batch, labels
#end of class DataSet


#使用skip-gram进行字向量建模
class CWord(object):
    #self.opt
    #slef.graph
    #self.valid_examples
    #self.sess
    #self.init
    #self.dataset
    #self.final_embddings

    #初始化函数
    def __init__(self, options, dataset):
        self.opt=options
        self.graph=tf.Graph()
        self.valid_examples = np.random.choice(self.opt.valid_window, self.opt.valid_size, replace=False)
        self.build_graph()
        self.sess=tf.Session(graph=self.graph)
        self.dataset=dataset

    # 模型构建
    def build_graph(self):
        with self.graph.as_default():
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.opt.batch_size])
            # Label data.
            self.train_labels = tf.placeholder(tf.int32, shape=[self.opt.batch_size, 1])
            # 验证数据集
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            # 使用gpu和cpu的两种构建方式
            # Look up embeddings for inputs.
            # 构建字向量矩阵, 维度为vocabulary_size*embedding_size, 每行代表一个字向量
            # 使用高斯分布随机变量初始化
            if self.opt.use_gpu:
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.opt.vocabulary_size, self.opt.embedding_size], -1.0, 1.0))
                # 根据每个中心词抽取出的词向量, 使用lookup函数实现
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                # 构建nce_loss所需的权重矩阵和偏置矩阵
                self.nce_weights = tf.Variable(
                    tf.truncated_normal([self.opt.vocabulary_size, self.opt.embedding_size],
                                    stddev=1.0 / math.sqrt(self.opt.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.opt.vocabulary_size]))
            else:
                # cpu的实现版本
                with tf.device("/cpu:0"):
                    self.embeddings = tf.Variable(
                        tf.random_uniform([self.opt.vocabulary_size, self.opt.embedding_size], -1.0, 1.0))
                    self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                    self.nce_weights = tf.Variable(
                        tf.truncated_normal([self.opt.vocabulary_size, self.opt.embedding_size],
                                        stddev=1.0 / math.sqrt(self.opt.embedding_size)))
                    self.nce_biases = tf.Variable(tf.zeros([self.opt.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # 构建loss, 使用nce函数, 利用负样本采样的方法
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.train_labels,
                               inputs=self.embed,
                               num_sampled=self.opt.num_sampled,
                               num_classes=self.opt.vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            # 利用梯度下降法求取最优矩阵参数
            # learn_rate设定学习速率, 不宜过大, 设置为1时在测试中误差有明显的发散发生, 缺省为0.5
            self.optimizer = tf.train.GradientDescentOptimizer(self.opt.learn_rate).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            # 将词向量矩阵归一化
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / self.norm
            # 利用验证数据集取出词向量
            # 与词向量的转置相乘,如果相似, 结果会更加靠近1
            # 后面使用argsort排序,就可以获得与验证数据集最相近的一组词向量
            self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
            #matmul是矩阵乘法
            self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            # 初始化所有的变量
            self.init = tf.global_variables_initializer()
            # 模型存储
            self.saver=tf.train.Saver([self.embeddings, self.nce_weights, self.nce_biases])
        
    # 模型训练
    def train(self):
        # 变量初始化
        with self.sess.as_default():
            self.init.run()
            print('Initialized')
        average_loss = 0
        #迭代次数
        step=0
        #迭代循环
        while True :
            step=step+1
            #从数据集中产生输入的批次, 输入给占位符
            batch_inputs, batch_labels = self.dataset.generate_batch(self.opt.batch_size, self.opt.num_skips, self.opt.skip_window)
            # 如果训练批次返回为空, 说明训练文件遍历完成, 退出循环
            if type(batch_inputs) == type(None) :
                break
            # 通过占位符输入训练数据
            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # 梯度下降,最小化loss
            _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            # 每2000次训练迭代输出一次中间结果
            if step % 2000 == 0:
                if step > 0:
                  # 计算平均偏差
                  average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                #sim = similarity.eval()
                sim=self.sess.run(self.similarity)
                for i in xrange(self.opt.valid_size):
                    valid_word = self.dataset.reverse_dictionary[self.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    #取出最相近的8个词
                    #使用sim结果的负值, 用argsort排序,得到结果最大的一组索引值
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                      #通过reverse_dictionary得到单词
                      close_word = self.dataset.reverse_dictionary[nearest[k]]
                      log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        #计算归一化的词向量矩阵
        #final_embeddings = normalized_embeddings.eval()
        self.final_embeddings = self.sess.run(self.normalized_embeddings)
        #保存训练好的字变量
        self.saver.save(self.sess, self.opt.modelPath+'/embed.ckpt')

    # 保存模型
    def save_final_embddings(self):
        np.save(self.opt.modelPath+"/final_embeddings.npy",self.final_embeddings)

    # 使用散点图作图, 没有调试过
    def plot_with_labels(self, low_dim_embs, labels, filename='tsne.png'):
        # 数据行数应该大于等于标签的行数
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            # 作图后标注标签
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)

    def plot_sample_data(self):
        try:
            # pylint: disable=g-import-not-at-top
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            #tsne是sklearn中的一种降维算法, 降到2维
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
            # 作图500个词
            plot_only = 1000
            low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
            # 取出每个单词作为标签
            labels = [reverse_dictionary[i] for i in xrange(plot_only)]
            # 作图
            self.plot_with_labels(low_dim_embs, labels)
            plt.show()
        except ImportError:
          print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#end of class CWord


def main(unused_argv):
    #Step 0: Set options
    opt=Options()

    # Step 1: Read the data.
    # 下载部分另行构造，使用维基百科的语料构建
    # Read the data into a list of strings.
    # 将文本分解成为每个中文字后形成列表返回
    dataSet=DataSet(opt.dataPath, opt.modelPath)
    
    # Step 2: Build the dictionary and replace rare words with UNK token.
    dataSet.build_dict(opt.vocabulary_size)
    dataSet.save_dict()

    dataSet.get_next_dataset()
    #print('Most common words (+UNK)', dataSet.count[:100])
    #print('Sample data', dataSet.data[:10], [dataSet.reverse_dictionary[i] for i in dataSet.data[:10]])
    
    # Step 3: Function to generate a training batch for the skip-gram model.
    #测试训练批次的产生
    #batch, labels = dataSet.generate_batch(batch_size=8, num_skips=2, skip_window=1)
    #for i in range(8):
      #print(batch[i], dataSet.reverse_dictionary[batch[i]], '->', labels[i, 0], dataSet.reverse_dictionary[labels[i, 0]])
    
    # Step 4: Build and train a skip-gram model.
    cword=CWord(opt,dataSet)
    
    # Step 5: Begin training.
    cword.train()
    
    # Step 6: Visualize the embeddings.
    #cword.plot_sample_data()

    # Step 7: save embeddings
    cword.save_final_embddings()

if __name__ == "__main__":
    tf.app.run()


# end of file

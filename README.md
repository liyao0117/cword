### CWORD中文字向量

中文和英语等西方文字不同, 中文是以方块字为单位, 然后再组合成为词. 因此中文中的词非常多, 词向量的训练难度也比较大. 
因此训练一个以字为单位的中文字向量, 会成为以中文为主的自然语言处理的一项基础工作.

#### 训练样本预处理

训练样本采用维基百科的中文语料, 可以参考[维基百科简体中文语料的获取](http://licstar.net/archives/262)
* 下载维基百科语料, 下载页面 [Wikimedia Downloads](https://dumps.wikimedia.org/)
进入"Database backup dumps", 选择"zhwiki", 其中会有下载文件链接

'''sh
wget https://dumps.wikimedia.org/zhwiki/20170920/zhwiki-20170920-pages-meta-current1.xml.bz2
wget https://dumps.wikimedia.org/zhwiki/20170920/zhwiki-20170920-pages-meta-current2.xml.bz2
wget https://dumps.wikimedia.org/zhwiki/20170920/zhwiki-20170920-pages-meta-current3.xml.bz2
wget https://dumps.wikimedia.org/zhwiki/20170920/zhwiki-20170920-pages-meta-current4.xml.bz2
'''

* 使用Wikipedia Extractor抽取正文
Wikipedia Extractor 是意大利人用 Python 写的一个维基百科抽取器，使用非常方便。[链接地址](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor)

'''sh
../WikiExtractor.py -b 500M -o extract zhwiki-20170901-pages-meta-current1.xml.bz2
cd extract
find . -name 'wiki*' -exec ../../filterTxt.sh {} \;
'''

* 繁简转换, 已包含在脚本filterTxt.sh中
* 使用filterTxt.sh去除英文, 标点, 空白行, 空格和一些特殊字符.

* 使用spliter切分文件, 可选
'''sh
split -a 3 -l 1000000 wiki*
'''

#### 模型构建
使用程序cword2vec.py 生成中文字向量
'''sh
python cword2vec.py
--datapath=./data
--modelpath=./model
--vocabulary_size=5000
--batch_size=128
--embedding_size=128
--skip_window=1
--num_skips=2
--valid_size=16
--valid_window=100
--num_sampled=64
--use_gpu=True
--learn_rate=0.5
usage: cword2vec.py [-h] [--datapath DATAPATH] [--modelpath MODELPATH]
                    [--vocabulary_size VOCABULARY_SIZE]
                    [--batch_size BATCH_SIZE]
                    [--embedding_size EMBEDDING_SIZE]
                    [--skip_window SKIP_WINDOW] [--num_skips NUM_SKIPS]
                    [--valid_size VALID_SIZE] [--valid_window VALID_WINDOW]
                    [--num_sampled NUM_SAMPLED] [--use_gpu USE_GPU]
                    [--learn_rate LEARN_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Directory to Sample data
  --modelpath MODELPATH
                        Director to Save model data
  --vocabulary_size VOCABULARY_SIZE
                        The top N words size.
  --batch_size BATCH_SIZE
                        Train batch size.
  --embedding_size EMBEDDING_SIZE
                        The embedding dimension size.
  --skip_window SKIP_WINDOW
                        How many words to consider left and right
  --num_skips NUM_SKIPS
                        How many times to reuse an input to generate a label.
  --valid_size VALID_SIZE
                        Random set of words to evaluate similarity on.
  --valid_window VALID_WINDOW
                        Only pick dev samples in the head of the distribution.
  --num_sampled NUM_SAMPLED
                        Number of negative examples to sample.
  --use_gpu USE_GPU     Use gpu or cpu, True is use gpu
  --learn_rate LEARN_RATE
                        Initial learning rate.
'''

#### 结果输出和适用

在model目录中, 包括三个部分的输出
* final_embeddings.npy 是numpy格式的中文字矩阵, 使用numpy.load('final_embeddings.npy')来加载
* 两个json词典文件, 是词袋模型的词典对象的json保存, 词典键为中文常用字, 值为编码, 编码所对应的下标从fianl_embeddings中取出的切片即为字向量
* 其他是tensorflow保存文件


## English Response Generate 2018-11

#### 1.preprocess

prepare() 将数据保存为 (text1, text2) 格式，打乱后划分训练、测试集

#### 2.represent

add_flag() 添加控制符，shift() 对 text2 分别删去 bos、eos 得到 sent2、label

tokenize() 通过 sent1 和 flag_text2 建立词索引、构造 embed_mat

align() 对训练数据 sent1 头部，sent2、label 尾部，填充或截取为定长序列

#### 3.build

通过 rnn 构建对话生成模型，s2s 编码器返回最后状态 h1_n 为解码器初始状态

att 编码器返回所有状态、最后状态之前为词特征 h1，解码器返回所有状态 h2

Attend() 比较 h2_i 与 h1、对 h1 加权平均返回语境 c_i，h2_i 与 c_i 共同决定输出

#### 4.generate

predict() 先对输入进行编码、再通过采样或搜索进行解码，check() 忽略无效词


# coding: utf-8

# In[1]:

# 1. inference() ——尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求。
# 2. loss() ——往inference 图表中添加生成损失（loss）所需要的操作（ops）。
# 3. training() ——往损失图表中添加计算并应用梯度（gradients）所需的操作。
import math
import tensorflow as tf


# In[2]:

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# In[3]:

# 推理（Inference）

# inference()函数会尽可能地构建图表，做到返回包含了预测结果（output prediction）
# 的Tensor。
# 它接受图像占位符为输入，在此基础上借助ReLu(Rectified Linear Units) 激活函数，
# 构建一对完全连接层（layers），以及一个有着十个节点（node）、指明了输出logtis 模型
# 的线性层。
# 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都
# 将带有其前缀。

# 在定义的作用域中，每一层所使用的权重和偏差都在tf.Variable 实例中生成，并
# 且包含了各自期望的shape

# 例如，当这些层是在hidden1作用域下生成时，赋予权重变量的独特名称将会是"hidden1/weights"。
# 每个变量在构建时，都会获得初始化操作（initializer ops）。
# 在这种最常见的情况下，通过tf.truncated_normal 函数初始化权重变量，给赋予
# 的shape 则是一个二维tensor，其中第一个维度代表该层中权重变量所连接（connect
# from）的单元数量，第二个维度代表该层中权重变量所连接到的（connect to）单元数
# 量。对于名叫hidden1的第一层，相应的维度则是[IMAGE_PIXELS, hidden1_units]，因为权
# 重变量将图像输入连接到了hidden1层。tf.truncated_normal初始函数将根据所得到的均
# 值和标准差，生成一个随机分布。
# 然后，通过tf.zeros 函数初始化偏差变量（biases），确保所有偏差的起始值都是0，
# 而它们的shape 则是其在该层中所接到的（connect to）单元数量。
# 图表的三个主要操作，分别是两个tf.nn.relu 操作，它们中嵌入了隐藏层所需的tf.
# matmul ；以及logits 模型所需的另外一个tf.matmul。三者依次生成，各自的tf.Variable实
# 例则与输入占位符或下一层的输出tensor 所连接。

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


# In[4]:

# 损失（Loss）

# loss()函数通过添加所需的损失操作，进一步构建图表。

# 首先，labels_placeholer中的值，将被编码为一个含有1-hot values 的Tensor。例如，
# 如果类标识符为“3”，那么该值就会被转换为： [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# 之后，又添加一个tf.nn.softmax_cross_entropy_with_logits 操作3，用来比较inference
# ()函数与1-hot 标签所输出的logits Tensor。

# 然后，使用tf.reduce_mean 函数，计算batch 维度（第一维度）下交叉熵（cross entropy）
# 的平均值，将将该值作为总损失。

# 最后，程序会返回包含了损失值的Tensor。

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


# In[5]:

# 训练

# training()函数添加了通过梯度下降（gradient descent）将损失最小化所需的操作。
# 首先，该函数从loss()函数中获取损失Tensor，将其交给[tf.scalar_summary] ，后
# 者在与SummaryWriter（见下文）配合使用时，可以向事件文件（events file）中生成汇总
# 值（summary values）。在本篇教程中，每次写入汇总值时，它都会释放损失Tensor 的
# 当前值（snapshot value）。

# 接下来，我们实例化一个[tf.train.GradientDescentOptimizer] ，负责按照所要求的
# 学习效率（learning rate）应用梯度下降法（gradients）

# 之后，我们生成一个变量用于保存全局训练步骤（global training step）的数值，并
# 使用minimize() 函数更新系统中的三角权重（triangle weights）、增加全局步骤的操作。
# 根据惯例，这个操作被称为train_op，是TensorFlow 会话（session）诱发一个完整训练
# 步骤所必须运行的操作

# 最后，程序返回包含了训练操作（training op）输出结果的Tensor。

def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


# In[6]:

# 评估模型

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


# In[ ]:




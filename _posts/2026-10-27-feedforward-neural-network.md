---
title: 前馈神经网络设计与训练
date: 2026-10-27 00:00:00 +0800
categories: [机器学习, 神经网络]
tags: [python, neural-network, fashion-mnist]
math: true
toc: true
---

本文配套代码：[ZZY0256/learn_fnn](https://github.com/ZZY0256/learn_fnn)。

针对 Fashion-MNIST 数据集中服装图片的分类问题，设计实现前馈神经网络并基于Python完成训练。

每张图像是 28x28 的灰度图，展平后得到 784 维输入向量，并将像素值从 [0,255] 归一化至 [-1.1]。

$$
x_i = \frac{\mathrm{pixel}_i}{127.5}-1,\quad x \in \mathbb{R}^{784}
$$

网络包含两个隐藏层和一个输出层：

$$
784 \rightarrow 128 \rightarrow 64 \rightarrow 10
$$

隐藏层通过 ReLU 激活函数，输出层通过 Softmax 得到 10个类别的概率。

## 参数定义与初始化

各层参数维度如下：

|     层     |                 权重矩阵                  |            偏置向量            |
| :--------: | :---------------------------------------: | :----------------------------: |
| 第一隐藏层 | $$W^{(1)} \in \mathbb{R}^{128 \times 784}$$ | $$b^{(1)} \in \mathbb{R}^{128}$$ |
| 第二隐藏层 | $$W^{(2)} \in \mathbb{R}^{64 \times 128}$$  | $$b^{(2)} \in \mathbb{R}^{64}$$  |
|   输出层   |  $$W^{(3)} \in \mathbb{R}^{10 \times 64}$$  | $$b^{(3)} \in \mathbb{R}^{10}$$  |

对于输入维度为 $$n$$ 的层，权重和偏置采用均匀随机初始化：

$$
W_{kj},b_k \sim U(-\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}) 
$$


## 前向传播

令 $$a^{(0)}=x$$，其中 $$z^{(l)}$$ 表示第 $$l$$ 层的线性输出，$$a^{(l)}$$ 表示激活后的输出。

两个隐藏层的计算为：

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)},\quad l=1,2
$$


$$
a^{(l)} = \mathrm{ReLU}(z^{(l)}),\quad \mathrm{ReLU}(u)=\max(0,u)
$$


输出层不使用 ReLU：

$$
z^{(3)} = W^{(3)}a^{(2)} + b^{(3)}
$$

通过 Softmax 将输出分数转换为类别概率：

$$
p_k = \frac{e^{z_k^{(3)}}}{\sum_{j=0}^9 e^{z_j^{(3)}}}
$$

计算时，为避免指数溢出，令 $$c = \max_j z_j^{(3)}$$，使用等价形式：

$$
p_k=\frac{e^{z_k^{(3)}-c}}{\sum_{j=0}^9 e^{z_j^{(3)}-c}}
$$

最终模型预测的类别为：

$$
\hat y = \arg \max_k p_k
$$


## 损失函数

采用交叉熵衡量预测概率和真实标签之间的差异，对于真实类别为 $$y$$ 的单个样本：

$$
\ell = - \log p_y
$$

代入 Softmax 并展开：

$$
\ell = - \log \frac{e^{z_y^{(3)}}}{\sum_j e^{z_j^{(3)}}} = \log (\sum_j e^{z_j^{(3)}}) - z_y^{(3)}
$$

为避免指数溢出，采用以下步骤转化成等价形式，令 $$c = \max_j z_j^{(3)}$$，对于每个 $$z_j^{(3)}$$，有：

$$
z_j^{(3)} = c + (z_j^{(3)} - c)
$$

因此：

$$
e^{z_j^{(3)}} = e ^ {c + (z_j^{(3)}-c)} = e^c e^{z_j^{(3)}-c}
$$

对所有类别求和：

$$
\sum_j e^{z_j^{(3)}} = e^{c} \sum_j e^{z_j^{(3)}-c}
$$

代回损失公式并展开对数：

$$
\ell = \log (e^c \sum_j e^{z_j^{(3)}-c})-z_y^{(3)}=\log e^c + \log(\sum_j e^{z_j^{(3)}-c}) - z_y^{(3)} = c + \log(\sum_j e^{z_j^{(3)}-c}) -z_y^{(3)}
$$

最终得到数值稳定形式：

$$
\ell = (c-z_y^{(3)}) + \log (\sum_j e^{z_j^{(3)}-c})
$$

对于包含 $$B$$ 个样本的批次，训练指标关注平均损失：

$$
L = \frac1{B} \sum_{s=1}^{B} \ell_s
$$


## 反向传播与梯度计算

反向传播利用链式法则，从输出层逐层计算参数梯度，对于单个样本的梯度，定义：

$$
\delta^{(l)} = \frac{\partial \ell}{\partial z^{(l)}}
$$


### 输出层梯度

令 $$q$$ 为真实类别的 one-hot 向量，即正确类别位置为1，其余为0，对交叉熵求导：

$$
\frac{\partial \ell}{\partial z_k^{(3)}} = \frac{e^{z_k^{(3)}}}{\sum_j e^{z_j^{(3)}}}-q_k=p_k-q_k
$$

因此：

$$
\delta^{(3)} = p - q
$$


### 隐藏层梯度

层 $$l$$ 的激活输出 $$a^{(l)} = \mathrm{ReLU}(z^{(l)})$$ 作为下一层的输入，满足：

$$
z_l^{(l+1)} = \sum_j W_{kj} a_j^{(l)} + b_k^{(l+1)}
$$

$$a_j^{(l)}$$ 会影响下一层的全部神经元，根据多元函数的链式法则：

$$
\frac{\partial \ell}{\partial a_j^{(l)}} = \sum_k \frac{\partial l}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial a_j^{(l)}} = \sum_k \delta_k^{(l+1)}W_{kj}^{(l+1)}
$$

由 ReLU 函数求导：

$$
\delta_j^{(l)} = \frac{\partial \ell}{\partial a_j^{(l)}}\frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} = (\sum_k W_{kj}^{(l+1)}\delta_k^{(l+1)}) \mathbf1[z_j^{(l)}>0]
$$

整理为向量形式：

$$
\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \mathbf1[z^{(l)}>0],\quad l=L-1,\ldots,1
$$

其中，$$\odot$$ 表示逐元素相乘；$$\mathbf1[z>0]$$ 在 $$z>0$$ 时取 1，否则取 0。本模型约定 ReLU 在 0 处的导数为 0。

### 权重与偏置梯度

由：

$$
z_k^{(l)} = \sum_j W_{kj} a_j^{(l-1)}+b_k^{(l)}
$$

根据链式法则：

$$
\frac{\partial \ell}{\partial W_{kj}^{(l)}} = \frac{\partial \ell}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial W_{kj}^{(l)}}=\delta_k^{(l)}a_j^{(l-1)}
$$

矩阵形式为：

$$
\frac{\partial \ell}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T,\quad \frac{\partial  \ell}{\partial b^{(l)}} = \delta^{(l)}
$$

对批次内各样本的梯度求平均，得到任意参数 $$\theta$$ 的批次梯度：

$$
\frac{\partial L}{\partial \theta} = \frac1{B} \sum_{s=1}^{B}\frac{\partial \ell_s}{\partial \theta}
$$


## 梯度下降与参数更新

反向传播计算出损失对各参数的梯度，优化器利用梯度调整参数，以降低损失。

用 $$\theta$$ 表示网络中任意一个权重或偏置，第 $$t$$ 次更新时，其梯度为：

$$
g_t = \frac{\partial L_t}{\partial \theta}
$$

其中 $$L_t$$ 是第 $$t$$ 个训练批次的平均损失，随后，沿梯度的反方向更新参数：

$$
\theta_t = \theta_{t-1} - \eta g_t
$$

其中 $$\eta \gt 0$$ 为学习率，用于控制更新步长。

## Adam 参数更新

普通下降直接使用当前批次的梯度，由于不同批次的梯度可能波动，而且不同参数的梯度尺度不同，可以使用Adam改善更新过程。每个权重，每个偏置，都有自己独立的两个状态 $$m$$ 和 $$v$$，分别代表参数的一阶矩和二阶矩。

在训练开始，尚未更新参数时，对每个参数设置：

$$
m_0 = 0,\quad v_0 = 0
$$

对于每一批次的训练，Adam用指数移动平均估计：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

其中，$$\beta_1$$ 控制保留多少历史信息，常取 0.9：

$$
m_t = 0.9 m_{t-1} + 0.1 g_t
$$

二阶矩同样采用指数移动平均：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

其中，$$\beta_2$$ 常取 0.999:

$$
v_t =0.999 v_{t-1} + 0.001 g_t^2
$$

由于最开始 $$m_0 = v_0 = 0$$，早期的移动平均会收到初始零值影响，因此，Adam用以下公式修正：

$$
\hat m = \frac{m_t}{1-\beta_1^t},\quad \hat v = \frac{v_t}{1-\beta_2^t}
$$

分母分别是截至第 $$t$$ 次更新，两个移动平均中已累积的权重总和。

用该修正值更新参数：

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
$$

其中，$$\eta$$ 是学习率，常用设置为 0.001，$$\varepsilon$$ 是防止分母为0的很小的正数，通常取 $$10^{-8}$$。

## 训练与评估流程

每个训练批次依次执行：

1. 前向传播，计算类别概率与损失。
2. 反向传播，累加各样本的参数梯度。
3. 将梯度除以批次样本数，得到平均梯度。
4. 使用 Adam 更新一次参数。

完整遍历训练集一次称为一个 epoch，数据集平均损失与分类准确率分别为：

$$
\bar L = \frac1{N}\sum_{s=1}^N \ell_s
$$


$$
\mathrm{Accuracy} = \frac1{N} \sum_{s=1}^N \mathbf1[\hat y_s = y_s]
$$


其中 $$N$$ 为评估数据集的样本总数。

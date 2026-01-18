## 0) 记号定义（先统一口径）

* 你要训练的学生模型：( f_\theta(x) \in \mathbb{R}^3 ) 输出 VAD（或你想要的更多维也行）
* 你现成的老师模型（vad-bert）：( f_T(x) )（只对那 6 种语言可靠）
* 一条多语平行样本：

  * (x^{(l)})：语言 (l) 的句子（(l\in \mathcal{L}_{src})，即那 6 语）
  * (x^{(zh)})：对应中文翻译（同一句意义）

---

## 1) 最核心的训练目标：三件事一起做

你想同时达成：

1. **中文对齐到老师空间**（让中文也有“像老师一样的 VAD”）
2. **跨语言一致性**（同一句不同语言，VAD 应该接近）
3. **旧语言不漂移**（学生在 6 语上的输出尽量保持老师原样）

这三件事对应三项 loss：

---

### (A) 中文蒸馏损失（Chinese distillation）

用平行对的“源语言老师输出”作为中文监督：

[
\mathcal{L}*{zh} = \mathbb{E}*{(x^{(l)},x^{(zh)})} \left[\left| f_\theta(x^{(zh)}) - f_T(x^{(l)}) \right|_2^2\right]
]

直觉：中文句子应该落到与“同义的源语言句子”相同的 VAD 坐标。

---

### (B) 跨语言一致性损失（language-invariance）

同义句的 VAD 要一致（学生自己约束自己）：

[
\mathcal{L}*{align} = \mathbb{E}*{(x^{(l)},x^{(zh)})} \left[\left| f_\theta(x^{(l)}) - f_\theta(x^{(zh)}) \right|_2^2\right]
]

这项能显著提升“中文/多语同空间”的稳定性，也会让中文学得更快。

---

### (C) 旧语言保持损失（retain old languages / anti-forgetting）

让学生在 6 语上的输出贴着老师，不要越训越歪：

[
\mathcal{L}*{retain} = \mathbb{E}*{x^{(l)}} \left[\left| f_\theta(x^{(l)}) - f_T(x^{(l)}) \right|_2^2\right]
]

这就是“保证原来支持语言尽量不受影响”的硬约束。

---

## 2) 总损失函数（你要的“落地公式”）

把三项加权：

[
\mathcal{L}*{total} =
\lambda*{zh}\mathcal{L}*{zh}+
\lambda*{align}\mathcal{L}*{align}+
\lambda*{retain}\mathcal{L}_{retain}
]

**推荐一个非常能跑起来的权重起点：**

* (\lambda_{zh}=1.0)
* (\lambda_{align}=0.5)
* (\lambda_{retain}=1.0)

如果你发现旧语言开始漂（验证集上 MSE 变大），就把 (\lambda_{retain}) 往上加到 2~4；
如果中文学不动，就把 (\lambda_{zh}) 加大，或降低 (\lambda_{retain})。

---

## 3) 训练数据怎么组 batch（工程关键）

每个 batch 里同时放两类样本：

* **平行对样本**：((x^{(l)},x^{(zh)})) 用来算 (\mathcal{L}*{zh}) 和 (\mathcal{L}*{align})
* **旧语言回放样本**：只取 (x^{(l)}) 用来算 (\mathcal{L}_{retain})

一个很好用的比例起点：

* 50% 平行对
* 50% 旧语言回放（从 OpenSubtitles 直接采 6 语句子即可）

> 这其实就是“replay + distillation”的经典抗遗忘套路，只不过监督信号来自老师 VAD。

---

## 4) 你说“6 语→中文”怎么用得更赚？

你可以把每条语义单元做成一个小集合：

[
{x^{(l_1)}, x^{(l_2)}, ..., x^{(l_k)}, x^{(zh)}}
]

然后扩展一致性损失为“所有语言互相拉近”：

[
\mathcal{L}*{align} =
\mathbb{E}\left[\sum*{i}\left|f_\theta(x^{(l_i)})-f_\theta(x^{(zh)})\right|_2^2\right]
]

甚至可以做“中心点”版本（更稳）：
[
c=\frac{1}{k}\sum_i f_\theta(x^{(l_i)}),\quad
\mathcal{L}*{align}=\left|f*\theta(x^{(zh)})-c\right|_2^2
]

这样中文不是跟某一种语言绑死，而是跟“多语共识”对齐。

---

## 5) 会踩的两个坑（提前帮你堵）

### 坑 1：字幕平行不严格

OpenSubtitles 有时不是逐句严格对齐。解决：

* 先用老师在 6 语上算 VAD
* 丢掉那些“同义对 VAD 差太大”的样本
  比如只保留：
  [
  \left| f_T(x^{(l)}) - f_T(x^{(l')}) \right|_2 < \tau
  ]
  （(\tau) 你可以先取 0.15~0.25，看输出范围）

### 坑 2：中文句子太口语/省略，导致 VAD 学偏

解决：在 batch 里混入新闻/TED 这种更“规范”中文（当稳定锚点），或者做个简单长度过滤（过短的“啊？”“行”先丢一丢）。

---

## 6) 应该怎么评估“旧语言没坏 + 中文变好了”

做两个验证集：

1. **旧语言保持分数**（越小越好）
   [
   E_{old}=\mathbb{E}*{x^{(l)}}\left[|f*\theta(x^{(l)})-f_T(x^{(l)})|_2\right]
   ]

2. **中文对齐分数**（越小越好）
   [
   E_{zh}=\mathbb{E}*{(x^{(l)},x^{(zh)})}\left[|f*\theta(x^{(zh)})-f_T(x^{(l)})|_2\right]
   ]

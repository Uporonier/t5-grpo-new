# GRPO-DDRO

## 简介
GRPO-DDRO是一个基于生成式强化学习的文档检索优化框架。该框架旨在通过生成式模型来优化文档检索的效果，提升检索系统的性能。

## 安装

### 加速解码过程
在生成式文档检索中，生成合法的docid是一个关键步骤。一般来说，生成docid需要满足一定的格式和约束条件。为了确保生成的docid合法，我们采用了限制性解码技术（constrained decoding）。这种技术通过在解码过程中引入约束，确保生成的序列符合预定义的规则。

限制性解码技术虽然保证了生成docid的合法性，但是也会导致生成效率的下降。为了提升生成效率，我们使用C++实现了trie树结构，并通过pybind11将其绑定到Python中，从而加速解码过程。

安装pybind11：
```bash
pip install pybind11
```

编译trie树C++代码：
```cpp
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)
```

##当前版本  token级别advantages

1.修改了采样
```python
        generation_kwargs={
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": args.num_generations,
            "temperature": 1.0, # 增加随机性，鼓励探索 (可以尝试 0.7 - 1.0)
            "top_k": 50, # 限制采样范围，防止生成太离谱的内容
            "top_p": 0.95,
        },    
```


2.全局奖励加到了每个token上
```python
  seq_rewards[t] = step_r + r_global
```


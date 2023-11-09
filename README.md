# LLM-Learning
大模型学习资料整理

# 1. 大模型相关论文

- BERT 论文：[Attention is all you need](https://arxiv.org/abs/1706.03762)
- 42篇精选大模型（LLM）论文及开源代码合集 - 鱼子酱的文章 - 知乎
https://zhuanlan.zhihu.com/p/653059045

# 2. 主流大模型

- OpenAI - ChatGPT
- Anthropic - Claude
- Meta
    - LLama https://github.com/facebookresearch/llama
    - MetaTransformer https://github.com/invictus717/MetaTransformer
    - CodeLlama（用来写代码的大模型）：https://github.com/facebookresearch/codellama
- 智谱 - https://github.com/THUDM/GLM
- 百川 - BaiChuan
- 阿里 - https://github.com/QwenLM/Qwen
- 上海人工智能实验室 -  https://github.com/InternLM/InternLM
- Stanford - https://github.com/tatsu-lab/stanford_alpaca
- AutoGPT:  An experimental open-source attempt to make GPT-4 fully autonomous. https://github.com/Significant-Gravitas/AutoGPT
- gpt4all: open-source LLM chatbots that you can run anywhere https://github.com/nomic-ai/gpt4all

# 3. 大模型训练 / 微调

- 大模型训练最主流技术栈 megatron-deepspeed:

https://github.com/microsoft/Megatron-DeepSpeed

- 微软的 LoRA 官方仓库：https://github.com/microsoft/LoRA
- PEFT 官方仓库：https://github.com/huggingface/peft
- Using Low-rank adaptation to quickly fine-tune diffusion models：https://github.com/cloneofsimo/lora
- 

# 4. 大模型推理

- NVidia 的 TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Flash-Attention 官方仓库：https://github.com/Dao-AILab/flash-attention
- UC Berkeley: https://github.com/FasterDecoding/Medusa
- FastLLM: 纯c++的全平台llm加速库，支持python调用，chatglm-6B级模型单卡可达10000+token / s，支持glm, llama, moss基座，手机端流畅运行 https://github.com/ztxz16/fastllm
- 

# 5. Diffusion / 生成扩散模型

- Stable-Diffusion 官方仓库：https://github.com/CompVis/stable-diffusion
- HuggingFace 的 SOTA Diffusion 模型库：

https://github.com/huggingface/diffusers

- Stabilitu-AI 生成模型库：https://github.com/Stability-AI/generative-models
- 

# 6. DL 框架 / DSL

- OpenAI -  https://github.com/openai/triton
- ML workload 调度
    - Ray: https://github.com/ray-project/ray
    - ColossalAI: https://github.com/hpcaitech/ColossalAI

# 7. 社区

- [HuggingFace 上的大模型资料合集](https://huggingface.co/docs/transformers/index)
- Llama 中文社区：https://github.com/FlagAlpha/Llama2-Chinese
- Numbers every LLM developers should know: https://github.com/ray-project/llm-numbers
- 

# 8. 课程

- Open AI 推荐的五门课程：
    - **EE263：《线性动力系统导论》**
    
    介绍应用线性代数和线性动力系统，并应用于电路，信号处理，通信和控制系统。课程网站：[EE263: Introduction to Linear Dynamical Systems (stanford.edu)](https://link.zhihu.com/?target=https%3A//ee263.stanford.edu/archive/)
    
    - **机器学习的神经网络**
    
    本课程将教你人工神经网络，以及如何将其应用于语音和物体识别，图像分割，建模语言，人体运动等。课程网站：[Lectures from the 2012 Coursera course:  Neural Networks for Machine Learning (toronto.edu)](https://link.zhihu.com/?target=https%3A//www.cs.toronto.edu/~hinton/coursera_lectures.html)
    
    - **CS231n：卷积神经网络用于视觉识别**
    
    本课程深入研究深度学习，专注于图像分类。在为期10周的课程中，学生将学习实施和训练他们的神经网络，并详细了解计算机视觉的前沿研究。课程网站：[CS231n: Deep Learning for Computer Vision](https://link.zhihu.com/?target=http%3A//cs231n.stanford.edu/)
    
    - **CS 287: Advanced Robotics by Pieter Abbeel**
    
    本课程将教你最先进的机器人系统下的数学和算法。这些技术中的大多数主要基于优化和概率推理。课程网站：[CS287 Fall 2019](https://link.zhihu.com/?target=https%3A//people.eecs.berkeley.edu/~pabbeel/cs287-fa19)
    
    - **CS 294: Deep Reinforcement Learning by Sergey Levine, John Schulman, Chelsea Finn**
    
    本课程涵盖监督学习、基本强化学习、高级模型学习和预测、蒸馏、奖励学习和高级深度强化学习。课程网站： [CS 294 Deep Reinforcement Learning, Spring 2017](https://link.zhihu.com/?target=https%3A//rll.berkeley.edu/deeprlcoursesp17)
    
- [Coursera: Generative AI with Large Language Models  - by Deeplearning.ai](https://www.coursera.org/learn/generative-ai-with-llms/)

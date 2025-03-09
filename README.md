# Biologically-Inspired Optimization Algorithms (BFOA)

## English Description

This repository contains implementations of biologically-inspired optimization algorithms for hyperparameter tuning of machine learning models and function optimization. The project explores various metaheuristic algorithms inspired by natural phenomena.

### Key Files

1. **bfoa_optimize.ipynb**
   - Implementation of the Bacterial Foraging Optimization Algorithm (BFOA)
   - Tests the algorithm on various benchmark functions (Rastrigin, Ackley, Booth, Levi, Easom, Schwefel, Himmelblau)
   - Simulates bacterial behaviors: chemotaxis, reproduction, elimination, and dispersal

2. **gwo_coevo_firework_optimize_w&b.py**
   - Implements Grey Wolf Optimizer (GWO) for hyperparameter tuning of Random Forest classifiers
   - Enhances GWO with co-evolutionary strategies and firework mechanisms
   - Compares performance between default Random Forest, standard GWO, and enhanced GWO
   - Integrates with Weights & Biases (wandb) for experiment tracking

### Implemented Algorithms

1. **Bacterial Foraging Optimization Algorithm (BFOA)**
   - Models the social foraging behavior of E. coli bacteria
   - Optimizes continuous functions through simulated bacterial movement

2. **Grey Wolf Optimizer (GWO)**
   - Standard implementation mimicking the leadership hierarchy of grey wolves
   - Used for hyperparameter optimization of machine learning models

3. **Enhanced GWO with Co-evolution and Firework Mechanism**
   - Improves GWO by dividing the population into evolving subgroups
   - Incorporates explosion-like operations to generate new solutions
   - Enhances exploration-exploitation balance for better convergence

## 中文描述

本仓库包含了用于机器学习模型超参数调优和函数优化的生物启发优化算法实现。项目探索了受自然现象启发的各种元启发式算法。

### 主要文件

1. **bfoa_optimize.ipynb**
   - 实现了细菌觅食优化算法 (BFOA)
   - 在多种基准函数上测试算法性能（Rastrigin、Ackley、Booth、Levi、Easom、Schwefel、Himmelblau）
   - 模拟细菌行为：趋化性、繁殖、消除和扩散

2. **gwo_coevo_firework_optimize_w&b.py**
   - 实现灰狼优化器 (GWO) 用于随机森林分类器的超参数调优
   - 通过协同进化策略和烟花机制增强GWO
   - 比较默认随机森林、标准GWO和增强型GWO的性能
   - 集成Weights & Biases (wandb)进行实验跟踪

### 已实现的算法

1. **细菌觅食优化算法 (BFOA)**
   - 模拟大肠杆菌的社会觅食行为
   - 通过模拟细菌运动优化连续函数

2. **灰狼优化器 (GWO)**
   - 标准实现，模拟灰狼的领导层次结构
   - 用于机器学习模型的超参数优化

3. **增强型GWO与协同进化和烟花机制**
   - 通过将种群分为进化子群组改进GWO
   - 结合爆炸式操作生成新解
   - 增强探索-利用平衡以获得更好的收敛性
# 代謝物 (metabolites) 於液相層析管柱 (Liquid Chromatography, LC) 中的遲滯時間 (retention time, RT) 預測

## Introduction

今日代謝體學 (metabolomics)學資料分析面臨兩個主要的挑戰：

1. 原始資料預處理 (Raw Data Preprocessing)
2. 代謝物辨識 (Metabolites identification)

前者在 xcms 等工具及其輔助的工具如參數預估工具、圖形化界面的出現後已經得到一定程度的改善，然而後者仍然是一個具有相當挑戰性的任務。隨著各種先進的質譜儀技術的發展，我們現在能夠獲得高精確度的小分子荷質比 (m/z)。然而，僅依靠荷質比來識別代謝物仍然不夠充分。因為一代謝物分子可能存在多種同分異構物（isomer）。同分異構物具有相同的分子質量，但卻有不同的生物特性。錯誤地將一個代謝物標識為另一個同分異構物，可能會導研究者對資料做出不正確的解讀。因此，引入分子在液相層析管柱中的遲滯時間 (retention time, RT) 數據對於正確識別代謝物至關重要，這提供了額外的信息，有助於更準確地進行代謝物辨識。

由於代謝物分子遲滯時間會受到管柱種類、沖提液梯度等多重因素影響，各實驗室一直以來都需要根據他們的具體需求和實驗條件購買標準品，建立專屬的遲滯時間資料庫。然而，受限於經費，這類由個別實驗室建立的資料庫通常包含不到千筆資料，且不會任意的公開。直到包含 80,038 個小分子在相同管柱條件下的遲滯時間資料的資料集 [METLIN small molecule retention time (SMRT)](https://www.nature.com/articles/s41467-019-13680-7) 的公佈，才有可能以深度學習這類較複雜的模型預測小分子於管柱中遲滯時間。

本實驗旨在利用以 SMRT 作為資料集訓練一圖神經網路（Graph Neural Network，GNN）模型預測小分子於管柱中的遲滯時間。我們使用 [RDKit](https://rdkit.org/) 以標準簡化分子線性表示式（smiles）作為輸入，計算小分子的分子、原子和化學鍵的特性。並將這些特性被用作圖特徵 (Graph Features)、端點特徵 (Node Features) 及 邊特徵(Edge Features)，最後將這些資訊作為圖神經網路的輸入，並以相應的小分子遲滯時間作為輸出，對該圖神經網路進行訓練。

## Goal

以圖深度學習 (Graph Neural Newwork, GNN) 建立一端到端模型 (End-to-End model)

- 輸入
  - 標準簡化分子線性表示式 （canonical smiles）
  - 利用標準簡化分子線性表示式列舉法 (smiles Enumeration) 產生的簡化分子線性表示式 (資料增強(Data Augmentation)
- 輸出
  - 小分子遲滯時間 (SMRT_gcn_att_pooling.py)
  - 沖提液比例 (SMRT_gcn_att_pooling_ratio.py)

## Packages version

- pytorch v1.10
- pyg v2.0.3
- rdkit v2020.09.5
- cuda v11.3.1

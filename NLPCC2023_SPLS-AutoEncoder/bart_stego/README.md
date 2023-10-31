# [Enhancing Semantic Consistency in Linguistic Steganography via Denosing Auto-Encoder and Semantic-Constrained Huffman Coding](https://github.com/Y-NLP/LinguisticSteganography/tree/main/NLPCC2023_SPLS-AutoEncoder#enhancing-semantic-consistency-in-linguistic-steganography-via-denosing-auto-encoder-and-semantic-constrained-huffman-coding)

The codes of our NLPCC2023 paper: "Enhancing Semantic Consistency in Linguistic Steganography via Denosing Auto-Encoder and Semantic-Constrained Huffman Coding".

# Dependency

- pytorch 1.3.1 
- 源码安装transformers 4.5.0
- You need to download the **bart-base-chinese-cluecorpussmall**  model () from [transformers library](https://huggingface.co/uer/bart-base-chinese-cluecorpussmall)

# Datasets

We put all two datasets mentioned in the paepr into the `datasets/` folder.


# Included Implementations

1. bart_huffman_bartemb.py`: implementations of method  **semantic huffman coding**  in the paper.

# How to run

1. 源码安装transformers

```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

2. 将transformers-main\src\transformers\models\bart\modeling_bart.py 替换为当前目录下的modeling_bart.py
3. 运行bart_huffman_bartemb.py

## Codes Reference

huffman 编码的实现参考自https://github.com/mickeysjm/StegaText

# Cite

```
@inproceedings{wang2023enhancing,
  title={Enhancing Semantic Consistency in Linguistic Steganography via Denosing Auto-Encoder and Semantic-Constrained Huffman Coding},
  author={Wang, Shuoxin and Li, Fanxiao and Yu, Jiong and Lai, Haosen and Wu, Sixing and Zhou, Wei},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={799--812},
  year={2023},
  organization={Springer}
}
```
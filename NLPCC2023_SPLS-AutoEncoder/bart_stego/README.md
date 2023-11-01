# [Enhancing Semantic Consistency in Linguistic Steganography via Denosing Auto-Encoder and Semantic-Constrained Huffman Coding](https://link.springer.com/chapter/10.1007/978-3-031-44696-2_62)

The codes of our NLPCC2023 paper: "Enhancing Semantic Consistency in Linguistic Steganography via Denosing Auto-Encoder and Semantic-Constrained Huffman Coding".

# Dependency

- pytorch 1.13.1 
- transformers 4.35.0.dev0 

# Datasets

We put all two datasets mentioned in the paepr into the `datasets/` folder.

# How to run

1. You need to install **transformers 4.35.0.dev0** from the source code.

```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

2. Replace transformers-main/src/transformers/models/bart/modeling_bart.py with the modeling_bart.py file in the current directory.
3. Download  model **bart-base-chinese-cluecorpussmall**   from [transformers library](https://huggingface.co/uer/bart-base-chinese-cluecorpussmall) and put the model into the model/uer-bart folder
4. run bart_huffman_bartemb.py

# Codes Reference

The implementation of huffman.py is based on the reference from   https://github.com/mickeysjm/StegaText."

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
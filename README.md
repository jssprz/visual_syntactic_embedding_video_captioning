# Improving Video Captioning with Temporal Composition of a Visual-Syntactic Embedding

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 
![Video Captioning and DeepLearning](https://img.shields.io/badge/VideoCaptioning-DeepLearning-orange)
![Source code of a WACV'21 paper](https://img.shields.io/badge/WACVpaper-SourceCode-yellow)
![MIT License](https://img.shields.io/badge/license-MIT-green)

This repository is the source code for the paper titled ***Improving Video Captioning with Temporal Composition of a Visual-Syntactic Embedding***.
Video captioning is the task of predicting a semantic and syntactically correct sequence of words given some context video. In this paper, we consider syntactic representation learning as an essential component of video captioning. We construct a visual-syntactic embedding by mapping into a common vector space a visual representation, that depends only on the video, with a syntactic representation that depends only on Part-of-Speech (POS) tagging structures of the video description. We integrate this joint representation into an encoder-decoder architecture that we call *Visual-Semantic-Syntactic Aligned Network (SemSynAN)*, which guides the decoder (text generation stage) by aligning temporal compositions of visual, semantic, and syntactic representations. We tested our proposed architecture obtaining state-of-the-art results on two widely used video captioning datasets: the Microsoft Video Description (MSVD) dataset and the Microsoft Research Video-to-Text (MSR-VTT) dataset.

## <a name="citation"></a>Citation

```
@article{PerezMartin2020AttentiveCaptioning,
	title={Improving Video Captioning with Temporal Composition of a Visual-Syntactic Embedding},
	author={Jesus Perez-Martin and Benjamin Bustos and Jorge Pérez},
	booktitle={IEEE Winter Conference on Applications of Computer Vision},
	year={2021}
}
```

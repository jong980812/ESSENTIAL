---
layout: project_page
permalink: /

title: "ESSENTIAL: Episodic and Semantic Memory Integration for Video Class-Incremental Learning"
authors:
  - '<a href="https://jong980812.github.io/" target="_blank">Jongseo Lee<sup>1*</sup></a>'
  - '<a href="https://github.com/Backdrop9019" target="_blank">Kyungho Bae<sup>2*</sup></a>'
  # - '<a href="" target="_blank">Kyle Min<sup>3</sup></a>'
  - Kyle Min<sup>3</sup>
  - '<a href="https://gyeongmoon.github.io/" target="_blank">Gyeong-Moon Park<sup>4†</sup></a>'
  - '<a href="https://sites.google.com/site/jchoivision/" target="_blank">Jinwoo Choi<sup>1†</sup></a>'
affiliations:
  - <sup>1</sup>Kyung Hee University
  - <sup>2</sup>Danggeun Market Inc.
  - <sup>3</sup>Intel Labs
  - <sup>4</sup>Korea University

# emails: 
#   - {jong980812, kyungho.bae, jinwoochoi}@khu.ac.kr
#   - kyle.min@intel.com
#   - gm-park@korea.ac.kr
paper: https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf
# video: https://www.youtube.com/results?search_query=turing+machine
code: https://github.com/topics/turing-machines
# data: https://huggingface.co/docs/datasets
highlight: "ICCV 2025 Highlight Paper"
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
In this work, we tackle the problem of video class-incremental learning (VCIL). Many existing VCIL methods mitigate catastrophic forgetting by rehearsal training with a few temporally dense samples stored in episodic memory, which is memory-inefficient. Alternatively, some methods store temporally sparse samples, sacrificing essential temporal information and thereby resulting in inferior performance. To address this trade-off between memory-efficiency and performance, we propose EpiSodic and SEmaNTIc memory integrAtion for video class-incremental Learning(ESSENTIAL). ESSENTIAL consists of episodic memory for storing temporally sparse features and semantic memory for storing general knowledge represented by learnable prompts. We introduce a novel memory retrieval (MR) module that integrates episodic memory and semantic prompts through cross-attention, enabling the retrieval of temporally dense features from temporally sparse features. We rigorously validate ESSENTIAL on diverse datasets: UCF-101, HMDB51, and Something-Something-V2 from the TCD benchmark and UCF-101, ActivityNet, and Kinetics-400 from the vCLIMB benchmark. Remarkably, with significantly reduced memory, ESSENTIAL achieves favorable performance on the benchmarks.
        </div>
    </div>
</div>

---
<br>
<br>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 0px;">
  
  <figure style="flex: 0 0 55%; text-align: center;">
    <img src="static/image/teaser.png" alt="Teaser Figure" style="width: 100%;">
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 5px;">
      1. Performance-memory plot on the UCF-101 dataset from the TCD benchmark
    </figcaption>
  </figure>

  <figure style="flex: 0 0 44%; text-align: center;">
    <img src="static/image/motivation.png" alt="Motivation Figure" style="width: 100%;">
    <figcaption style="font-size: 0.9em; color: gray; margin-top: 5px;">
      2. MR module achieves a better performance-memory-efficiency trade-off.
    </figcaption>
  </figure>

</div>



## Motivation

ESSENTIAL is designed to overcome the trade-off in VCIL between **performance** and **memory-efficiency**. In Figure2, 

- 📉 **(a) Temporally dense features** stored in episodic memory yield high performance but suffer from **low memory-efficiency**.  
- 💾 **(b) Temporally sparse features** improve **memory-efficiency**, but lack of temporal context leads to **performance degradation**.  
- ⚖️ **(c) ESSENTIAL** combines the best of both: storing **temporally sparse features** and lightweight **semantic prompts** to maintain efficiency, while the **MR module** integrates episodic and semantic memory to reconstruct **temporally dense features**.

> The distance between the **retrieved feature vector** and the **original temporally dense feature vector** is significantly smaller than that between the **temporally sparse feature vector** and the **temporally dense feature vector**.

*By effectively retrieving temporal information from temporally sparse features, the MR module enables a high memory-efficiency-performance trade-off as demonstrated in **Figure 1**.*



## Architecture
1. Turing first presented the concept of a "computable number," which refers to a number that can be computed by an algorithm or a definite step-by-step process.
2. He introduced the notion of a Turing machine, an abstract computational device consisting of an infinite tape divided into cells and a read-write head. The machine can read and write symbols on the tape, move the head left or right, and transition between states based on a set of rules.
3. Turing demonstrated that the set of computable numbers is enumerable, meaning it can be listed in a systematic way, even though it is not necessarily countable.
4. He proved the existence of non-computable numbers, which cannot be computed by any Turing machine.
5. Turing showed that the Entscheidungsproblem is undecidable, meaning there is no algorithm that can determine, for any given mathematical statement, whether it is provable or not.

![Turing Machine](static/image/Training.png)


## Table: Comparison of Computable and Non-Computable Numbers

| Computable Numbers | Non-Computable Numbers |
|-------------------|-----------------------|
| Rational numbers, e.g., 1/2, 3/4 | Transcendental numbers, e.g., π, e |
| Algebraic numbers, e.g., √2, ∛3 | Non-algebraic numbers, e.g., √2 + √3 |
| Numbers with finite decimal representations | Numbers with infinite, non-repeating decimal representations |

He used the concept of a universal Turing machine to prove that the set of computable functions is recursively enumerable, meaning it can be listed by an algorithm.

## Significance
Turing's paper laid the foundation for the theory of computation and had a profound impact on the development of computer science. The Turing machine became a fundamental concept in theoretical computer science, serving as a theoretical model for studying the limits and capabilities of computation. Turing's work also influenced the development of programming languages, algorithms, and the design of modern computers.

## Citation
```
@article{turing1936computable,
  title={On computable numbers, with an application to the Entscheidungsproblem},
  author={Turing, Alan Mathison},
  journal={Journal of Mathematics},
  volume={58},
  number={345-363},
  pages={5},
  year={1936}
}
```

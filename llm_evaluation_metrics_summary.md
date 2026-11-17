# Comprehensive Summary: LLM Performance Evaluation Metrics

This document summarizes the presentations covering core evaluation metrics for Large Language Models (LLMs) and Natural Language Processing (NLP) tasks from the Department of Applied Mathematics and Statistics (ITC).

## 1. Exact Match (EM) — Group 1

* **Core Concept:** A strict, binary evaluation metric that checks whether a model's generated prediction is completely identical to the reference ground-truth answer ($1$ for an exact match, $0$ otherwise).

* **Formula:**
  

  $$
  \text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(P_i = R_i)
  $$

* **Normalization:** Strict EM is sensitive to casing, punctuation, and articles. **Normalized EM** applies preprocessing (lowercasing, removing punctuation/articles, trimming whitespace) prior to comparison.

* **Best Use Cases:** Short-answer Question Answering (QA), math problems, and tasks requiring a single exact factual entity.

* **Pros & Cons:**

  * **Pros:** Highly objective, fast, reproducible, and leaves no room for ambiguity.

  * **Cons:** Overly strict; gives zero partial credit and penalizes valid paraphrases or minor formatting differences.

## 2. Accuracy — Group 2

* **Core Concept:** Measures the overall proportion of correct predictions made by the model out of all total predictions.

* **Formula:**
  

  $$
  \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

* **Implementation:** Easily computed manually or via `sklearn.metrics.accuracy_score`.

* **Best Use Cases:** Balanced binary classification (e.g., Yes/No, Sentiment Analysis) and Multiple-Choice Question Answering (MCQA).

* **Pros & Cons:**

  * **Pros:** Intuitive, mathematically simple, and provides a quick high-level snapshot of performance.

  * **Cons:** Highly misleading on **imbalanced datasets** and does not distinguish between different types of errors (False Positives vs. False Negatives).

## 3. Precision, Recall, and F1-Score — Group 3

* **Core Concept:** Complementary classification and token-overlap metrics derived from a confusion matrix ($TP$, $FP$, $FN$, $TN$).

* **Formulas:**

  * **Precision** (reliability of positive predictions):
    

    $$
    \text{Precision} = \frac{TP}{TP + FP}
    $$

  * **Recall** (coverage of actual positive cases):
    

    $$
    \text{Recall} = \frac{TP}{TP + FN}
    $$

  * **F1-Score** (harmonic mean balancing Precision and Recall):
    

    $$
    \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    $$

* **Best Use Cases:** Imbalanced classification, spam detection (prioritizing Precision), medical screening (prioritizing Recall), and token-level QA evaluation.

* **Pros & Cons:**

  * **Pros:** Captures error trade-offs effectively; F1 prevents high scores when either Precision or Recall is poor.

  * **Cons:** Does not directly account for True Negatives ($TN$) in F1, ignores overall language fluency, and relies heavily on accurate ground-truth labels.

## 4. BLEU Score — Group 4

* **Core Concept:** *Bilingual Evaluation Understudy* — measures surface-level lexical similarity between a candidate output and reference text(s) using **clipped n-gram precision** (typically 1-grams to 4-grams) combined with a **Brevity Penalty (BP)** to discourage overly short outputs.

* **Formula:**
  

  $$
  \text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \ln p_n\right)
  $$

  
  where $BP = 1$ if $c \ge r$, and $BP = e^{1 - r/c}$ if $c < r$ ($c$ = candidate length, $r$ = reference length).

* **Best Use Cases:** Machine Translation and corpus-level text generation benchmarks.

* **Pros & Cons:**

  * **Pros:** Fast, inexpensive, repeatable, and rewards matching local word order via higher-order n-grams.

  * **Cons:** Misses synonyms and valid paraphrases, fails to measure factual correctness, and performs poorly on isolated short sentences.

## 5. ROUGE Score — Group 5

* **Core Concept:** *Recall-Oriented Understudy for Gisting Evaluation* — measures how much important information (content coverage) from a reference text is captured in a generated summary.

* **Key Variants:**

  * **ROUGE-N:** Overlap of consecutive $N$-grams (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).

  * **ROUGE-L:** Longest Common Subsequence (LCS), allowing gaps while preserving word order.

  * **ROUGE-Lsum:** Summary-level union LCS across sentence alignments.

  * **ROUGE-W:** Weighted LCS that rewards consecutive matching runs over scattered matches.

  * **ROUGE-S / ROUGE-SU:** Skip-bigram overlap (ordered pairs with gaps) and skip-bigrams plus unigrams.

* **Best Use Cases:** Text summarization and gisting tasks.

* **Pros & Cons:**

  * **Pros:** Effective at measuring content recall and information coverage across multiple granularities.

  * **Cons:** Lacks semantic understanding (misses synonyms), cannot detect factual contradictions, and raw recall favors overly verbose outputs.

## 6. METEOR Score — Group 6

* **Core Concept:** *Metric for Evaluation of Translation with Explicit ORdering* — improves upon BLEU by aligning candidate and reference words across three stages (**Exact**, **Stem**, and **Synonym** matching via WordNet) and applying a fragmentation penalty for disrupted word order.

* **Formula:**
  

  $$
  \text{METEOR} = F_{\text{mean}} \times (1 - \text{Penalty})
  $$

  
  where $F_{\text{mean}} = \frac{10PR}{R + 9P}$ (heavily weighting Recall) and $\text{Penalty} = 0.5 \times \left(\frac{\text{chunks}}{m}\right)^3$.

* **Implementation:** Available in Python via `nltk.translate.meteor_score`.

* **Best Use Cases:** Machine translation, paraphrasing, and short-answer evaluation where lexical variation occurs.

* **Pros & Cons:**

  * **Pros:** Recognizes stems and synonyms, balances precision and recall, and explicitly penalizes scrambled word order.

  * **Cons:** Relies on external linguistic resources (like WordNet), varies across languages, and cannot verify factual truth.

## 7. BERTScore — Group 7

* **Core Concept:** Evaluates **semantic similarity** (meaning) rather than exact string overlap by computing pairwise cosine similarity between contextual token embeddings from a pretrained transformer (e.g., BERT, RoBERTa).

* **Methodology:**

  1. Tokenize and embed reference ($x$) and candidate ($\hat{x}$) using contextual embeddings.

  2. Compute a cosine similarity matrix ($S_{i,j} = \mathbf{x}_i^\top \hat{\mathbf{x}}_j$).

  3. Perform greedy matching to compute $R_{\text{BERT}}$ (row maxima mean), $P_{\text{BERT}}$ (column maxima mean), and $F_{\text{BERT}}$.

  4. Apply optional IDF weighting and baseline rescaling.

* **Implementation:** Available via the `bert-score` Python library.

* **Best Use Cases:** Abstractive summarization, paraphrasing, open-ended QA, and machine translation.

* **Pros & Cons:**

  * **Pros:** Accurately credits valid paraphrases and captures context-dependent word meanings.

  * **Cons:** High computational cost (requires neural network inference), sensitive to backbone model/layer selection, and suffers from **negation blindness** (missing reversed meanings).

## 8. Word Error Rate (WER) — Group 8

* **Core Concept:** The standard edit-distance metric for Automatic Speech Recognition (ASR), measuring word-level Levenshtein distance between a reference transcript and a model hypothesis.

* **Formula:**
  

  $$
  \text{WER} = \frac{S + D + I}{N}
  $$

  
  where $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, and $N$ = total words in the reference. *(Lower is better;* $0\%$ *is perfect; can exceed* $100\%$*)*.

* **Implementation:** Computed in Python using `jiwer.wer(ref, hyp)`.

* **Best Use Cases:** Speech-to-text transcription and ASR model benchmarking (e.g., Whisper-base vs. Whisper-tiny).

* **Pros & Cons:**

  * **Pros:** Industry standard, breaks down exact error types ($S/D/I$), and produces a single comparable score.

  * **Cons:** Treats all word errors equally regardless of semantic impact (e.g., minor pluralization vs. critical negation) and penalizes valid synonyms.

## 9. Character Error Rate (CER) — Group 9

* **Core Concept:** Analogous to WER, CER measures the minimum number of single-character edits required to transform a reference text into the predicted text, divided by the total number of reference characters.

* **Formula:**
  

  $$
  \text{CER} = \frac{S + D + I}{N}
  $$

  
  where $S$, $D$, $I$ are character-level substitutions, deletions, and insertions, and $N$ is the reference character count.

* **Implementation:** Computed in Python using `jiwer.cer(reference, prediction)`.

* **Best Use Cases:** Optical Character Recognition (OCR), spelling correction, non-space-delimited languages (e.g., Khmer), and fixed-format text/codes.

* **Pros & Cons:**

  * **Pros:** Does not require word tokenization/splitting and catches fine-grained spelling or character recognition mistakes.

  * **Cons:** Completely ignores semantic meaning and is sensitive to whitespace and punctuation cleanup.

## 10. Comparative Evaluation Overview — Group 10

Group 10 synthesizes all nine metric topics into a unified framework across four evaluation perspectives:

### Metric Taxonomy & Task Mapping

| Perspective | Metrics | Core Signal | Recommended NLP / LLM Tasks | Primary Blind Spot | 
| ----- | ----- | ----- | ----- | ----- | 
| **1. Exact / Label-Based** | **EM, Accuracy, P/R/F1** | Exact string or class label match | Extractive QA, Classification, MCQA | Rejects paraphrases; hides class imbalance (Accuracy) | 
| **2. Lexical Similarity** | **BLEU, ROUGE, METEOR** | N-gram overlap, LCS, & stem/synonym alignment | Machine Translation, Summarization | High overlap $\neq$ factual or semantic correctness | 
| **3. Semantic Similarity** | **BERTScore** | Contextual token-embedding cosine similarity | Open-ended generation, Paraphrasing, Summarization | High compute cost; may miss negation/contradictions | 
| **4. Edit-Based Errors** | **WER, CER** | Word-level or character-level edit distance | Speech Recognition (ASR), OCR | Ignores meaning; sensitive to segmentation/punctuation | 

### Key Takeaways

1. **No Universally Best Metric:** Metric selection must match the output format and evaluation goal.

2. **Similarity Is Not Factuality:** High lexical or embedding similarity does not guarantee truthful or safe outputs.

3. **Use Complementary Metrics:** Combine fast lexical/edit metrics with semantic metrics (e.g., ROUGE + BERTScore) and human judgment for reliable LLM evaluation.
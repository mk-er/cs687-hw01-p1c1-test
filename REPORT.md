# Homework 1 · Project 1, Checkpoint 1 — Report

Name:
Student number:
Date:

---

## 1 · Fertility table

Paste the table printed by `python scripts/fertility.py`.

```text
(paste here)
```

## 2 · Reflection (two sentences)

What do these numbers imply about the effective context length and the cost per
query for a Turkish-speaking user of an English-centric model?

> (your two sentences here)

## 3 · Three strings where GPT-2 uses fewer tokens than your tokenizer

From `python scripts/inspect_merges.py`. For each one, say in a single sentence
why the difference arises, in terms of what each tokenizer saw during training.

| string | GPT-2 tokens | your tokens | why |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

## 4 · Comprehension questions

Answer these in your own words.

**Question A.** Your tokenizer was trained with 500 merges. Predict what would
change if you doubled that to 1000: (i) the average number of tokens per word,
(ii) the number of parameters an embedding matrix would need, and (iii) how
well the rarest tokens would be trained. Which effect do you expect to dominate
for a corpus of this size, and why?

>

**Question B.** Your `encode` method applies merge rules in the order they were
learned. Explain what would go wrong if it instead applied whichever applicable
merge was most frequent in the training corpus. Why would this bug be difficult
to notice?

>

**Question C.** The pre-tokenization step keeps merges from crossing word
boundaries. Give one reason this is desirable, and name the tokenization
artifact from the notes that it directly causes.

>

**Question D.** Suppose your model reaches a loss of 4.1 nats per token using
your 500-merge tokenizer. Using the numbers your own fertility experiment
produced, convert this into bits per byte. Show every step of the conversion.

>

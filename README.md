# CS 687 · Homework 1 · Project 1, Checkpoint 1

> **TEST REPOSITORY:** This public repository exists only to rehearse the
> student Colab and submission workflow. It is not an announced course release.

This is a standalone student repository for Homework 1. It introduces
byte-level byte-pair encoding, next-token training windows, embeddings, and the
units used to report language-model loss.

You do not need another CS 687 code repository to complete this homework. A
graphics card is not required.

## What you must complete

The Colab notebook contains ten assessed answer cells:

- four short guided-calculation cells;
- two programming-task cells; and
- four Markdown report-response cells.

The programming tasks are:

| task | tagged notebook cell | function or class |
|---|---|---|
| Task 1 | `answer-task-1` | `BPETokenizer.train` |
| Task 2 | `answer-task-2` | `NextTokenDataset.__init__` |

All six code-answer cells contain `NotImplementedError` placeholders. The
surrounding code is supplied complete. The four report cells contain written
response placeholders. Your completed notebook is the authoritative submission;
you do not need to copy its implementations into `.py` files.

## Grading

Homework 1 is graded out of 100 points:

| assessed work | points |
|---|---:|
| Four guided calculations | 16 |
| Task 1: BPE training | 14 |
| Task 2: sliding-window dataset | 10 |
| Four written responses | 60 |
| **Total** | **100** |

All six code exercises are graded all-or-nothing. Each guided calculation
receives either 4 points or 0 points. Task 1 receives its 14 points only if it
passes every staff-controlled Task 1 test, and Task 2 receives its 10 points
only if it passes every staff-controlled Task 2 test. Partially correct code
does not receive partial credit. The written responses are graded separately.

## Google Colab

Open `notebooks/homework01_colab.ipynb` in Google Colab and run it from top to
bottom. Its setup cell clones this repository and installs the required Python
packages. This rehearsal copy clones the fixed `v0.15-test` release from the
`mk-er` test repository.

The notebook lets you develop both implementations interactively. A focused
check follows each code-answer cell so that you receive feedback before
continuing. Near the end, a final cell runs all 35 public tests together. Run
these cells inside the notebook: a separate `pytest` process cannot see
definitions that exist only in the notebook kernel.

## Homework workflow

1. Read the Lecture 1 notes.
2. Work through the notebook in order. Complete all four guided-calculation
   cells and the two programming-task cells, and run each immediate check.
3. Restart the runtime and run the complete notebook from top to bottom.
4. Confirm that the final cell repeats all four calculation checks and reports
   all 35 programming tests passing.
5. Complete the four questions in the notebook's report section,
   using evidence from your notebook run.

Do not modify the tests to make an implementation pass. The tests describe the
required behavior and grading uses a staff-controlled copy.

## Deliverables

- the completed Homework 1 notebook

The completed notebook is the single Homework 1 submission. Before uploading
it to Moodle, rename it to `homework01_colab_STUDENTNUMBER.ipynb`, replacing
`STUDENTNUMBER` with your student number; for example,
`homework01_colab_21802962.ipynb`. Moodle's account record,
rather than the filename alone, remains the authoritative student identity.

## Submission structure check

Before uploading, run the public structure checker on the exact notebook file
you intend to submit:

```text
python submission_check.py path/to/homework01_colab_STUDENTNUMBER.ipynb
```

If you work in Colab, first download the completed notebook and then run the
final notebook cell. It opens an upload dialog and runs the checker on the
`.ipynb` file you select. Select the exact downloaded file that you intend to
submit. The command above remains the equivalent route for local work.

Replace `STUDENTNUMBER` with your actual student number. Fix every reported
error before submitting. If the checker reports an older assignment version,
do not edit the version number yourself; contact the course staff for
inspection. The checker verifies only that the notebook can be processed by
the grading system. It does not grade the answers, replace the public tests, or
guarantee a particular mark.

An unreadable file or a file that is not an `.ipynb` notebook cannot be graded
and receives zero.

## Repository layout

```text
notebook_support.py  supplied setup and public-test helpers
submission_check.py supplied pre-submission structure checker
cs687/       supplied implementation modules used by the notebook
tests/       inspectable public tests run by the notebook
data/        the English–Turkish parallel corpus
notebooks/   the Colab entry point and authoritative code submission
```

# CS 687 · Homework 1 · Project 1, Checkpoint 1

> **TEST REPOSITORY:** This public repository exists only to rehearse the
> student Colab workflow. It is not an announced course release.

This is a standalone student repository for Homework 1. It introduces
byte-level byte-pair encoding, next-token training windows, embeddings, and the
units used to report language-model loss.

You do not need another CS 687 code repository to complete this homework. A
graphics card is not required.

## What you must implement

There are two incomplete tasks in the Colab notebook:

| task | tagged notebook cell | function or class |
|---|---|---|
| Task 1 | `answer-task-1` | `BPETokenizer.train` |
| Task 2 | `answer-task-2` | `NextTokenDataset.__init__` |

The answer cells contain `NotImplementedError` placeholders. The surrounding
code is supplied complete. Your completed notebook is the authoritative code
submission; you do not need to copy its implementations into `.py` files.

## Recommended route: Google Colab

Open `notebooks/homework01_colab.ipynb` in Google Colab and run it from top to
bottom. Its setup cell clones this repository and installs the required Python
packages. This test copy clones the fixed `v0.10-test` release from `mk-er`.

The notebook lets you develop both implementations interactively. A focused
public-test cell follows each answer cell so that you receive feedback before
continuing. Near the end, a final cell runs all 31 public tests together. Run
these cells inside the notebook: a separate `pytest` process cannot see
definitions that exist only in the notebook kernel.

## Optional local route

Python 3.10 or later is required. From the repository root:

```text
python -m venv .venv
```

On Windows PowerShell, activate it with:

```text
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```text
source .venv/bin/activate
```

Then install the dependencies and open the notebook:

```text
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab notebooks/homework01_colab.ipynb
```

The notebook workflow is the same locally and in Colab. Complete the tagged
answer cells and use the notebook's public-test cell. The tests remain visible
under `tests/`, but running them in a new shell process tests the untouched
module skeletons rather than your in-memory notebook answers.

## Homework workflow

1. Read the Lecture 1 notes.
2. Complete Task 1 in the `answer-task-1` cell and work through the tokenizer
   investigations that follow it.
3. Complete Task 2 in the `answer-task-2` cell and work through the remaining
   notebook experiments.
4. Restart the runtime and run the complete notebook from top to bottom.
5. Confirm that the public-test cell reports all 31 tests passing.
6. Download the completed `.ipynb` file and complete the separate report
   template.

Do not modify the tests to make an implementation pass. The tests describe the
required behavior and grading uses a staff-controlled copy.

## Deliverables

- the completed Homework 1 notebook, including working implementations in both
  tagged answer cells;
- all 31 included tests passing;
- a separate completed report containing the fertility table and two-sentence
  reflection;
- three comparisons with the GPT-2 tokenizer; and
- answers to comprehension questions A–D in the report.

The notebook and report will be submitted through Moodle. Their exact filenames,
the final report format, and the upload procedure will be announced before
release. Do not submit the cloned repository, virtual environment, caches, or
generated Python bytecode.

## What happens in Homework 2

Homework 2 will be a new standalone repository containing the course's
canonical Homework 1 implementation. Your ability to begin Homework 2 will not
depend on carrying this checkout forward or on whether your implementation here
matches the canonical one.

## Repository layout

```text
cs687/       supplied implementation modules used by the notebook
tests/       inspectable public tests run by the notebook
scripts/     tokenizer inspection and fertility experiments
data/        the English–Turkish parallel corpus
notebooks/   the Colab entry point and authoritative code submission
REPORT.md    the current report template (final format still to be chosen)
```

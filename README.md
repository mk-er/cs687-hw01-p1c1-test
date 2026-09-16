# CS 687 · Homework 1 · Project 1, Checkpoint 1

> **TEST REPOSITORY:** This public repository exists only to rehearse the
> student Colab workflow. It is not an announced course release.

This is a standalone student repository for Homework 1. It introduces
byte-level byte-pair encoding, next-token training windows, embeddings, and the
units used to report language-model loss.

You do not need another CS 687 code repository to complete this homework. A
graphics card is not required.

## What you must implement

There are two incomplete tasks:

| task | file | function |
|---|---|---|
| Task 1 | `cs687/tokenizer.py` | `BPETokenizer.train` |
| Task 2 | `cs687/data.py` | `NextTokenDataset.__init__` |

The incomplete locations contain `TODO` comments and raise
`NotImplementedError`. The surrounding modules are supplied complete.

## Recommended route: Google Colab

Open `notebooks/homework01_colab.ipynb` in Google Colab and run it from top to
bottom. Its setup cell clones this repository and installs the required Python
packages. This test copy clones the fixed `v0.2-test` release from `mk-er`.

The notebook lets you develop the two functions interactively. Before
submitting, copy those implementations into `cs687/tokenizer.py` and
`cs687/data.py`, then run the repository tests.

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

Then install and test:

```text
python -m pip install -r requirements.txt
python -m pytest -q
```

The repository contains 31 tests. At the beginning, 24 failures in
`test_tokenizer.py` and `test_data.py` are expected because Tasks 1 and 2 are
incomplete. The seven tests for the supplied embedding and unit-conversion code
should already pass.

## Homework workflow

1. Read the Lecture 1 notes.
2. Complete Task 1 and run:

   ```text
   python -m pytest tests/test_tokenizer.py -q
   ```

3. Run the tokenizer investigations:

   ```text
   python scripts/inspect_merges.py
   python scripts/fertility.py
   ```

4. Complete Task 2 and run:

   ```text
   python -m pytest tests/test_data.py -q
   ```

5. Run the full included test suite:

   ```text
   python -m pytest -q
   ```

6. Complete `REPORT.md`.

Do not modify the tests to make an implementation pass. The tests describe the
required behavior and grading uses a staff-controlled copy.

## Deliverables

- working implementations of Tasks 1 and 2;
- all 31 included tests passing;
- the fertility table and two-sentence reflection;
- three comparisons with the GPT-2 tokenizer; and
- answers to comprehension questions A–D in `REPORT.md`.

The submission location and procedure will be announced separately. Do not
submit the virtual environment, caches, or generated Python bytecode.

## What happens in Homework 2

Homework 2 will be a new standalone repository containing the course's
canonical Homework 1 implementation. Your ability to begin Homework 2 will not
depend on carrying this checkout forward or on whether your implementation here
matches the canonical one.

## Repository layout

```text
cs687/       implementation files
tests/       public tests for this homework
scripts/     tokenizer inspection and fertility experiments
data/        the English–Turkish parallel corpus
notebooks/   the Colab entry point
REPORT.md    the Homework 1 report template
```

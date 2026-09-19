"""Operational helpers used by the Homework 1 student notebook.

The notebook keeps assessed code and conceptually relevant experiments visible.
This module contains only setup and public-test orchestration that students are
not expected to implement.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Type


def prepare_notebook(repo_root: Path) -> Path:
    """Enter the repository, install its requirements, and make it importable."""
    repo_root = repo_root.resolve()
    required = (repo_root / "cs687", repo_root / "tests", repo_root / "requirements.txt")
    if not required[0].is_dir() or not required[1].is_dir() or not required[2].is_file():
        raise RuntimeError(
            "The notebook setup did not locate a Homework 1 repository containing "
            "cs687/, tests/, and requirements.txt."
        )

    os.chdir(repo_root)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    print("ready")
    return repo_root


def check_saved_submission(
    upload_files: Callable[[], dict[str, bytes]] | None = None,
) -> None:
    """Upload and structurally check the exact notebook intended for submission.

    In Colab, calling this function with no arguments opens the browser upload
    dialog. The optional callback exists so the public workflow can be tested
    without depending on the Colab interface.
    """
    if upload_files is None:
        try:
            from google.colab import files
        except ImportError:
            print(
                "Local execution: save the completed notebook, then run:\n"
                "python submission_check.py "
                "path/to/homework01_colab_STUDENTNUMBER.ipynb"
            )
            return
        upload_files = files.upload

    uploaded = upload_files()
    notebook_names = [name for name in uploaded if name.lower().endswith(".ipynb")]
    if len(uploaded) != 1 or len(notebook_names) != 1:
        raise RuntimeError("Upload exactly one completed .ipynb notebook.")

    notebook_path = Path.cwd() / Path(notebook_names[0]).name
    checker_path = Path(__file__).resolve().with_name("submission_check.py")
    result = subprocess.run(
        [sys.executable, str(checker_path), str(notebook_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise AssertionError(
            "The saved notebook did not pass the structure check. "
            "Fix every message above, download it again, and rerun this cell."
        )
    print("Structure check passed. Submit this same file to Moodle.")


def _is_incomplete(function: Callable[..., object]) -> bool:
    """Return whether an answer still contains its supplied placeholder."""
    return "NotImplementedError" in function.__code__.co_names


def check_probability_calculation(
    probabilities: object,
    true_token_id: int,
    true_probability: object,
    loss_nats: object,
    token_perplexity: object,
) -> None:
    """Check the guided probability-to-loss calculation."""
    import torch

    expected_probability = probabilities[true_token_id]
    if not torch.isclose(true_probability, expected_probability):
        raise AssertionError(
            "Select the probability at true_token_id from the model's distribution."
        )
    expected_loss = -torch.log(expected_probability)
    if not torch.isclose(loss_nats, expected_loss):
        raise AssertionError(
            "Negative log-likelihood is -ln(p), where p is the probability "
            "assigned to the observed token."
        )
    if not torch.isclose(token_perplexity, torch.exp(expected_loss)):
        raise AssertionError("Perplexity is exp(loss_nats).")
    print("Probability, loss, and perplexity are consistent.")


def check_fertility_measure(measure: Callable[..., object]) -> None:
    """Check the two normalization formulas used by the fertility experiment."""
    class SevenTokenTokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return list(range(7))

    sentences = ["one two", "üç"]
    result = measure(SevenTokenTokenizer(), sentences)
    text = " ".join(sentences)
    expected_fertility = 7 / len(text.split())
    expected_tokens_per_byte = 7 / len(text.encode("utf-8"))

    if not isinstance(result, dict):
        raise AssertionError("measure must return a dictionary.")
    if "fertility" not in result or "tokens_per_byte" not in result:
        raise AssertionError(
            "Return both 'fertility' and 'tokens_per_byte' measurements."
        )
    if abs(result["fertility"] - expected_fertility) > 1e-12:
        raise AssertionError("Fertility is number of tokens divided by number of words.")
    if abs(result["tokens_per_byte"] - expected_tokens_per_byte) > 1e-12:
        raise AssertionError(
            "Tokens per byte is number of tokens divided by the UTF-8 byte count."
        )
    print("Both tokenizer-fragmentation measurements are correct.")


def check_embedding_components(
    token_vectors: object,
    position_vectors: object,
    combined: object,
    module_output: object,
) -> None:
    """Check the guided token-plus-position embedding calculation."""
    import torch

    if token_vectors.ndim != 3:
        raise AssertionError("Token vectors should have shape (B, T, d_model).")
    if position_vectors.ndim != 2:
        raise AssertionError("Position vectors should have shape (T, d_model).")
    expected = token_vectors + position_vectors
    if not torch.allclose(combined, expected):
        raise AssertionError(
            "Add each position vector to the token vectors at that position; "
            "PyTorch broadcasts across the batch dimension."
        )
    if not torch.allclose(module_output, expected):
        raise AssertionError(
            "The manual token-plus-position result should match InputEmbedding."
        )
    print("Token and position vectors reproduce the module output.")


def check_unit_conversion(
    loss_nats: float,
    tokens_per_word: float,
    bytes_per_word: float,
    tokens_per_byte: float,
    loss_bits: float,
    loss_bits_per_byte: float,
) -> None:
    """Check the guided conversion from nats/token to bits/byte."""
    import math

    expected_tokens_per_byte = tokens_per_word / bytes_per_word
    expected_loss_bits = loss_nats / math.log(2)
    expected_loss_bits_per_byte = expected_loss_bits * expected_tokens_per_byte

    if abs(tokens_per_byte - expected_tokens_per_byte) > 1e-12:
        raise AssertionError("tokens/byte = tokens/word divided by bytes/word.")
    if abs(loss_bits - expected_loss_bits) > 1e-12:
        raise AssertionError("bits/token = nats/token divided by ln(2).")
    if abs(loss_bits_per_byte - expected_loss_bits_per_byte) > 1e-12:
        raise AssertionError("bits/byte = bits/token multiplied by tokens/byte.")
    print("The unit conversion is correct and dimensionally consistent.")


def _run_pytest(path: str, failure_message: str) -> None:
    """Run one public-test target in the current notebook kernel."""
    import pytest

    result = pytest.main(["-q", "--tb=short", path])
    if result != pytest.ExitCode.OK:
        raise AssertionError(failure_message)


def run_task1_tests(train: Callable[..., object]) -> None:
    """Attach the Task 1 answer and run its public tokenizer tests."""
    if _is_incomplete(train):
        raise AssertionError(
            "Task 1 is still incomplete. Replace the NotImplementedError in the "
            "answer-task-1 cell, run that cell again, and rerun this test cell."
        )

    from cs687.tokenizer import BPETokenizer

    BPETokenizer.train = train
    _run_pytest(
        "tests/test_tokenizer.py",
        "Task 1 has not passed all public tests. Review the failures above.",
    )
    print("Task 1 passed all public tests.")


def _attach_task2(dataset_class: Type[object]) -> None:
    """Expose the notebook's Task 2 class to modules imported by the tests."""
    import cs687 as cs687_package
    import cs687.data as data_module

    cs687_package.NextTokenDataset = dataset_class
    data_module.NextTokenDataset = dataset_class


def run_task2_tests(dataset_class: Type[object]) -> None:
    """Attach the Task 2 answer and run its public dataset tests."""
    if _is_incomplete(dataset_class.__init__):
        raise AssertionError(
            "Task 2 is still incomplete. Replace the NotImplementedError in the "
            "answer-task-2 cell, run that cell again, and rerun this test cell."
        )

    _attach_task2(dataset_class)
    _run_pytest(
        "tests/test_data.py",
        "Task 2 has not passed all public tests. Review the failures above.",
    )
    print("Task 2 passed all public tests.")


def run_all_public_tests(
    train: Callable[..., object], dataset_class: Type[object]
) -> None:
    """Attach both notebook answers and run the complete public test suite."""
    incomplete = []
    if _is_incomplete(train):
        incomplete.append("Task 1")
    if _is_incomplete(dataset_class.__init__):
        incomplete.append("Task 2")
    if incomplete:
        raise AssertionError(
            "Complete and rerun these answer cells first: " + ", ".join(incomplete)
        )

    from cs687.tokenizer import BPETokenizer

    BPETokenizer.train = train
    _attach_task2(dataset_class)
    _run_pytest(
        "tests",
        "One or more public tests failed. Review the failures above.",
    )
    print("All 35 public tests passed.")

"""Check the interpreter version before anything else in the package runs.

This module exists because of a real failure.  The repository has always
required Python 3.10 or later — several modules annotate optional arguments as
``torch.Generator | None``, which is syntax that only exists from 3.10 — and
that requirement was written down in the README and enforced nowhere.  On an
older interpreter the result was fifteen collection errors reading

    TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

which says nothing about versions to anybody who has not met it before, and
which appears fifteen times so that the useful part scrolls off the screen.

The trap is easy to fall into and has nothing to do with skill: macOS still
ships Python 3.9 as ``python3``, so a student who types the obvious commands
gets the obvious interpreter and then a wall of red.

A requirement that is not checked is a requirement that will be discovered by
whoever is least equipped to diagnose it.  So it is checked here, at the top of
the package, where every path into the code — the tests, the scripts, the
notebooks, the demonstrations — has to pass through it.
"""

import sys

MINIMUM = (3, 10)


def check(minimum=MINIMUM):
    """Raise a message a person can act on, rather than one they must decode."""
    if sys.version_info >= minimum:
        return
    want = ".".join(str(p) for p in minimum)
    have = ".".join(str(p) for p in sys.version_info[:3])
    raise RuntimeError(
        "\n"
        "\n  This repository needs Python " + want + " or later. You are running "
        + have + ".\n"
        "\n  The interpreter in use is:\n"
        "    " + sys.executable + "\n"
        "\n  On macOS the system 'python3' is 3.9, which is old enough that some\n"
        "  of the type annotations in this code are syntax errors for it. This is\n"
        "  not something you have done wrong, and nothing in your own work needs\n"
        "  changing.\n"
        "\n  To fix it, install a newer Python and build the environment with that\n"
        "  one rather than with the system copy:\n"
        "\n    brew install python@3.12                  # or python.org's installer\n"
        "    /opt/homebrew/bin/python3.12 -m venv .venv\n"
        "    source .venv/bin/activate\n"
        "    python3 -V                                # must print 3.12.x\n"
        "    pip install -r requirements.txt\n"
        "\n  A virtual environment inherits whichever interpreter created it, so\n"
        "  making one with the system python3 and then upgrading pip inside it\n"
        "  does not help; the environment has to be created by the new one.\n"
    )

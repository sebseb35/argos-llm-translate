from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    if not verbose:
        # Third-party translation libs can be extremely chatty at INFO and dump
        # large token arrays, which can bloat terminal scrollback/memory.
        logging.getLogger("argostranslate").setLevel(logging.WARNING)
        logging.getLogger("stanza").setLevel(logging.WARNING)

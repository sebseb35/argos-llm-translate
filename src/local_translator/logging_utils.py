from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    if verbose:
        # Re-enable INFO logs if they were globally disabled by a previous run.
        logging.disable(logging.NOTSET)
    else:
        # Hard gate: suppress INFO/DEBUG records globally, even when third-party
        # loggers set their own level/handlers (argostranslate can be very chatty).
        logging.disable(logging.INFO)
        # Third-party translation libs can be extremely chatty at INFO and dump
        # large token arrays, which can bloat terminal scrollback/memory.
        logging.getLogger("argostranslate").setLevel(logging.WARNING)
        logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)
        logging.getLogger("stanza").setLevel(logging.WARNING)

"""Tinker training helpers.

Only ``raw_doc.py`` lives here now: raw-document SFT (continued-pretraining
style — tokenise whole documents, no chat template), used by exploration 03.

The astra ``conditional_misalignment`` chat-SFT + tracer pipeline that this
package used to carry (trainer / dataset_builder / spec / render / tracer_panel)
was removed — tracers are out of scope, and the live character-training SFT
(``explorations/04_.../scripts/train_sft.py``) drives cookbook's
``supervised.train`` + ``FromConversationFileBuilder`` directly rather than
through this layer. The one reusable piece worth keeping, the truncated-assistant
chat-SFT builder, was lifted to ``weird_personas.tinker_datasets.ChatSFTDatasetBuilder``.
See ``docs/src_overview.md``.

Public surface:
    raw_doc.run               — train a LoRA on raw documents (continued pretraining).
    raw_doc.RawDocDatasetBuilder — cookbook builder that renders docs → PrebuiltDataset.
    raw_doc.build_doc_datums  — tokenise documents into all-ones-weight datums.
"""

from . import raw_doc  # noqa: F401
from .raw_doc import RawDocDatasetBuilder, build_doc_datums  # noqa: F401
from .raw_doc import run as run_raw_doc  # noqa: F401

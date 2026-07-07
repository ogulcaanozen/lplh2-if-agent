"""Module 3: Experience Library.

Captures structured experience summaries, tracks event dedup state, and
retrieves relevant past experiences via the configured vector backend.
"""

import os
import logging
import hashlib
import json
import numbers
from . import config

logger = logging.getLogger(__name__)


class ExperienceLib:
    """Experience Library with RAG for reflective learning.

    When score changes (gain or loss/death), the system summarizes
    the interaction history into structured experience and stores it
    for retrieval. During gameplay, relevant experiences are retrieved
    using embedding similarity. Neutral-event keys are stored separately
    from the vector backend so dedup is Chroma-independent.
    """

    def __init__(self, persist_dir: str = None, event_index_dir: str = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.event_index_dir = event_index_dir or config.EXPERIENCE_INDEX_DIR
        self.experiences = []     # raw list of experience texts
        self._collection = None   # ChromaDB collection (lazy init)
        self._chroma_client = None
        self._event_index = None  # persistent event dedup index

    def reset(self):
        """Reset experience library for a fresh start."""
        self.experiences = []
        # Don't reset ChromaDB - experiences persist across epochs
        # (this is key to learning across epochs)
        # Don't reset the event index either; it prevents duplicate neutral
        # and score-gain summaries across epochs in the same experiment.

    def _init_chroma(self):
        """Lazily initialize ChromaDB."""
        if self._collection is not None:
            return

        os.makedirs(self.persist_dir, exist_ok=True)

        # Disable Chroma's anonymized telemetry before importing/initializing
        # the client. This avoids Colab/OpenTelemetry runtime conflicts.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

        import chromadb
        from chromadb.config import Settings

        self._chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized at {self.persist_dir} "
                     f"with {self._collection.count()} existing experiences")

    def store_experience(self, experience_text: str, metadata: dict = None):
        """Store an experience summary in the library.

        Args:
            experience_text: The structured experience summary from LLM
            metadata: Optional metadata (location, score, epoch, etc.)
        """
        if not experience_text or experience_text.strip() == "":
            return

        self.experiences.append(experience_text)

        # Store in ChromaDB for retrieval
        try:
            self._init_chroma()
            doc_id = hashlib.md5(experience_text.encode()).hexdigest()
            chroma_metadata = self._sanitize_chroma_metadata(metadata or {})

            self._collection.add(
                documents=[experience_text],
                ids=[doc_id],
                metadatas=[chroma_metadata],
            )
            logger.info(f"Stored experience (total: {len(self.experiences)})")
        except Exception as e:
            logger.warning(f"Failed to store experience in ChromaDB: {e}")
            # Still keep in memory list

    @staticmethod
    def _sanitize_chroma_metadata(metadata: dict) -> dict:
        """Return metadata in Chroma's scalar-only format.

        Chroma accepts only str/int/float/bool metadata values. The agent's
        internal metadata may contain None, lists, dicts, or NumPy-style scalar
        values from game/runtime APIs, so normalize before vector storage.
        """
        clean = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            key = str(key)
            if isinstance(value, bool):
                clean[key] = value
            elif isinstance(value, numbers.Integral):
                clean[key] = int(value)
            elif isinstance(value, numbers.Real):
                clean[key] = float(value)
            elif isinstance(value, str):
                clean[key] = value
            elif isinstance(value, (list, tuple, dict)):
                clean[key] = json.dumps(value, ensure_ascii=False)
            else:
                clean[key] = str(value)
        return clean

    def neutral_event_seen(self, event_key: str) -> bool:
        """True if a neutral-state event has already been summarized.

        The key is built from the triggering event rather than the LLM-written
        summary, so equivalent events are skipped even when summaries would be
        worded differently across epochs.
        """
        return self.event_seen(event_key)

    def record_neutral_event(self, event_key: str, metadata: dict = None):
        """Persist a neutral-state event key after its summary is stored."""
        self.record_event(event_key, metadata=metadata)

    def event_seen(self, event_key: str) -> bool:
        """True if an experience event key has already been recorded."""
        if not event_key:
            return False
        self._load_event_index()
        return event_key in self._event_index

    def record_event(self, event_key: str, metadata: dict = None):
        """Persist an event key after its summary decision is handled."""
        if not event_key:
            return
        self._load_event_index()
        self._event_index[event_key] = metadata or {}
        self._save_event_index()

    def _event_index_path(self) -> str:
        return os.path.join(self.event_index_dir, "experience_event_index.json")

    def _load_event_index(self):
        if self._event_index is not None:
            return
        os.makedirs(self.event_index_dir, exist_ok=True)
        path = self._event_index_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = data.get("events", {}) if isinstance(data, dict) else {}
            self._event_index = events if isinstance(events, dict) else {}
        except FileNotFoundError:
            self._event_index = {}
        except Exception as e:
            logger.warning(f"Failed to load experience event index: {e}")
            self._event_index = {}

    def _save_event_index(self):
        os.makedirs(self.event_index_dir, exist_ok=True)
        path = self._event_index_path()
        tmp_path = f"{path}.tmp"
        data = {
            "version": 1,
            "description": (
                "Experience event keys already summarized or intentionally "
                "skipped, including neutral events and score gains."
            ),
            "events": self._event_index or {},
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"Failed to save experience event index: {e}")

    def retrieve_relevant_structured(self, query: str, top_k: int = None,
                                     fetch_k: int = None,
                                     where: dict = None) -> list[dict]:
        """Retrieve relevant past experiences with metadata for richer prompt rendering."""
        top_k = top_k or config.EXPERIENCE_TOP_K
        fetch_k = fetch_k or max(config.EXPERIENCE_FETCH_K, top_k)
        try:
            self._init_chroma()
            if self._collection.count() == 0:
                return []

            query_kwargs = {
                "query_texts": [query],
                "n_results": min(fetch_k, self._collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where
            results = self._collection.query(**query_kwargs)
            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                metadatas = (results.get("metadatas") or [[]])[0] or []
                distances = (results.get("distances") or [[]])[0] or []
                output = []
                for i, doc in enumerate(docs):
                    output.append({
                        "text": doc,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else None,
                    })
                return output
        except Exception as e:
            logger.warning(f"ChromaDB structured retrieval failed: {e}")

        if not self.experiences:
            return []
        return [
            {
                "text": exp,
                "metadata": {"kind": "recent", "source": "memory_fallback"},
                "distance": None,
            }
            for exp in self.experiences[-fetch_k:]
        ]

    def retrieve_relevant(self, query: str, top_k: int = None) -> str:
        """Retrieve relevant past experiences using RAG.

        Args:
            query: Current game context to search against
            top_k: Number of experiences to retrieve

        Returns:
            Formatted string of relevant experiences
        """
        top_k = top_k or config.EXPERIENCE_TOP_K

        # Guard against empty DB — check ChromaDB, not the in-memory list.
        # self.experiences is cleared on reset() but ChromaDB persists across
        # epochs (that cross-epoch persistence is the paper's core learning mechanism).
        try:
            self._init_chroma()
            if self._collection.count() == 0:
                return "No relevant experiences found yet."

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
            )

            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                output = []
                for i, doc in enumerate(docs, 1):
                    output.append(f"Experience {i}:\n{doc}")
                return "\n\n".join(output)

        except Exception as e:
            logger.warning(f"ChromaDB retrieval failed: {e}")

        # Fallback: return most recent in-memory experiences (current session only)
        if not self.experiences:
            return "No relevant experiences found yet."
        recent = self.experiences[-top_k:]
        output = []
        for i, exp in enumerate(recent, 1):
            output.append(f"Experience {i} (recent):\n{exp}")
        return "\n\n".join(output)

    def clear_collection(self):
        """Clear all stored experiences (for completely fresh start)."""
        self.experiences = []
        self._event_index = {}
        try:
            self._init_chroma()
            self._chroma_client.delete_collection(config.CHROMA_COLLECTION)
            self._collection = None
            self._chroma_client = None
            try:
                os.remove(self._event_index_path())
            except FileNotFoundError:
                pass
            logger.info("Experience library cleared")
        except Exception as e:
            logger.warning(f"Failed to clear ChromaDB: {e}")

    def num_experiences(self) -> int:
        """Number of stored experiences (total across all epochs in ChromaDB)."""
        try:
            self._init_chroma()
            return self._collection.count()
        except Exception:
            return len(self.experiences)

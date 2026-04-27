import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Silence graphiti's noisy "index already exists" errors on rerun.
logging.getLogger("graphiti_core").setLevel(logging.CRITICAL)
logging.getLogger("neo4j").setLevel(logging.ERROR)
from cogram import Graphiti
from cogram.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from cogram.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from cogram.llm_client.config import LLMConfig
from cogram.llm_client.openai_generic_client import OpenAIGenericClient

load_dotenv()


@dataclass
class Settings:
    """Cogram runtime settings.

    Three LLM tiers are addressable independently so users can route the
    expensive extraction call to OpenAI/Anthropic and the cheaper post-write
    calls to DeepSeek/Qwen/Ollama:

      large_llm_*  — graphiti's entity/edge extraction (heavy multi-shot call)
      small_llm_*  — intent annotation, narration, profile distill, contradiction
                     classifier (cheap structured-output calls, fired in pipeline)
      embedder_*   — text-embedding for entities, episodes, edges, narratives

    For backward compat with v0.1 .env files:
      GRAPHITI_LLM_MODEL   falls into large_llm_model when LARGE_LLM_MODEL unset
      ANNOTATOR_LLM_MODEL  falls into small_llm_model when SMALL_LLM_MODEL unset
      EMBEDDING_MODEL      falls into embedder_model  when EMBEDDER_MODEL  unset
      OPENAI_API_KEY / LLM_BASE_URL  serve as the default for any tier whose
      LARGE_/SMALL_/EMBEDDER_ specific key/url is unset.

    The legacy attributes `api_key`, `base_url`, `graphiti_llm_model`,
    `annotator_llm_model`, `embedding_model` are preserved as aliases that
    point at the matching tiered field, so older code keeps working.
    """
    # Tiered model names
    large_llm_model: str
    small_llm_model: str
    embedder_model: str
    embedding_dim: int

    # Tiered endpoints
    large_llm_api_key: str
    large_llm_base_url: str
    small_llm_api_key: str
    small_llm_base_url: str
    embedder_api_key: str
    embedder_base_url: str

    # Cluster/global config
    rate_limit_per_min: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # ------------------------------------------------------------------
    # Backward-compat aliases (read-only; old call sites keep working)
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        """Legacy alias: defaults to the small-tier key (most call sites
        historically used this for annotator/narrator/profile)."""
        return self.small_llm_api_key

    @property
    def base_url(self) -> str:
        """Legacy alias: defaults to the small-tier endpoint."""
        return self.small_llm_base_url

    @property
    def graphiti_llm_model(self) -> str:
        """Legacy alias for the large-tier model (graphiti extraction)."""
        return self.large_llm_model

    @property
    def annotator_llm_model(self) -> str:
        """Legacy alias for the small-tier model (annotation/narration/profile)."""
        return self.small_llm_model

    @property
    def embedding_model(self) -> str:
        """Legacy alias for embedder_model."""
        return self.embedder_model

    @classmethod
    def from_env(cls) -> "Settings":
        # ------------------------------------------------------------------
        # Default key + base_url (used when a tier doesn't set its own)
        # ------------------------------------------------------------------
        default_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("NVIDIA_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )
        if not default_key or "replace" in default_key:
            raise SystemExit("Set OPENAI_API_KEY (or NVIDIA_API_KEY) in .env")

        default_base = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL")
            or "https://api.openai.com/v1"
        )

        # ------------------------------------------------------------------
        # Tiered model names — new env wins; old env is fallback
        # ------------------------------------------------------------------
        large_model = (
            os.environ.get("LARGE_LLM_MODEL")
            or os.environ.get("GRAPHITI_LLM_MODEL")
            or "gpt-4o-mini"
        )
        small_model = (
            os.environ.get("SMALL_LLM_MODEL")
            or os.environ.get("ANNOTATOR_LLM_MODEL")
            or "gpt-4o-mini"
        )
        embedder_model = (
            os.environ.get("EMBEDDER_MODEL")
            or os.environ.get("EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )

        # ------------------------------------------------------------------
        # Tiered endpoints — fall back to the shared default key/base_url
        # ------------------------------------------------------------------
        return cls(
            large_llm_model=large_model,
            small_llm_model=small_model,
            embedder_model=embedder_model,
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1536")),

            large_llm_api_key=os.environ.get("LARGE_LLM_API_KEY", "").strip() or default_key,
            large_llm_base_url=os.environ.get("LARGE_LLM_BASE_URL", "").strip() or default_base,
            small_llm_api_key=os.environ.get("SMALL_LLM_API_KEY", "").strip() or default_key,
            small_llm_base_url=os.environ.get("SMALL_LLM_BASE_URL", "").strip() or default_base,
            embedder_api_key=os.environ.get("EMBEDDER_API_KEY", "").strip() or default_key,
            embedder_base_url=os.environ.get("EMBEDDER_BASE_URL", "").strip() or default_base,

            rate_limit_per_min=int(os.environ.get("RATE_LIMIT_PER_MIN", "150")),
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
        )


def build_graphiti(settings: Settings | None = None) -> Graphiti:
    s = settings or Settings.from_env()

    # Graphiti's extraction is the heavy LLM call → routed to LARGE tier.
    # `small_model` is graphiti's secondary model field (used for cheaper
    # validation / dedup helpers); we route that to our SMALL tier.
    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=s.large_llm_api_key,
            model=s.large_llm_model,
            small_model=s.small_llm_model,
            base_url=s.large_llm_base_url,
        )
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=s.embedder_api_key,
            embedding_model=s.embedder_model,
            embedding_dim=s.embedding_dim,
            base_url=s.embedder_base_url,
        )
    )

    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            api_key=s.large_llm_api_key,
            model=s.large_llm_model,
            small_model=s.small_llm_model,
            base_url=s.large_llm_base_url,
        )
    )

    provider = os.environ.get("GRAPH_PROVIDER", "neo4j").lower()
    graph_kwargs: dict = {
        "llm_client": llm_client,
        "embedder": embedder,
        "cross_encoder": cross_encoder,
        "max_coroutines": 10,
    }

    if provider == "falkordb":
        from cogram.driver.falkordb_driver import FalkorDriver
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))
        graph_kwargs["graph_driver"] = FalkorDriver(host=host, port=port)
        g = Graphiti(**graph_kwargs)
    elif provider == "kuzu":
        from cogram.driver.kuzu_driver import KuzuDriver
        db_path = os.environ.get("KUZU_DB_PATH", "./cache/kuzu.db")
        graph_kwargs["graph_driver"] = KuzuDriver(db=db_path)
        g = Graphiti(**graph_kwargs)
    else:
        g = Graphiti(s.neo4j_uri, s.neo4j_user, s.neo4j_password, **graph_kwargs)

    from cogram.utils.rate_limit import patch_clients as _rate_patch  # noqa: WPS433
    from cogram.llm_client.engram import patch_clients as _cache_patch  # noqa: WPS433
    _rate_patch(g)
    _cache_patch(g)
    return g

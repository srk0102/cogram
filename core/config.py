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
    api_key: str
    base_url: str
    graphiti_llm_model: str
    annotator_llm_model: str
    embedding_model: str
    embedding_dim: int
    rate_limit_per_min: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    @classmethod
    def from_env(cls) -> "Settings":
        # Provider-agnostic key lookup; first non-empty wins
        key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("NVIDIA_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )
        if not key or "replace" in key:
            raise SystemExit("Set OPENAI_API_KEY (or NVIDIA_API_KEY) in .env")

        # Default to OpenAI; override with NVIDIA_BASE_URL or LLM_BASE_URL for NIM/etc.
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL")
            or "https://api.openai.com/v1"
        )

        return cls(
            api_key=key,
            base_url=base_url,
            graphiti_llm_model=os.environ.get("GRAPHITI_LLM_MODEL", "gpt-4o-mini"),
            annotator_llm_model=os.environ.get("ANNOTATOR_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1536")),
            rate_limit_per_min=int(os.environ.get("RATE_LIMIT_PER_MIN", "150")),
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
        )


def build_graphiti(settings: Settings | None = None) -> Graphiti:
    s = settings or Settings.from_env()

    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=s.api_key,
            model=s.graphiti_llm_model,
            small_model=s.graphiti_llm_model,
            base_url=s.base_url,
        )
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=s.api_key,
            embedding_model=s.embedding_model,
            embedding_dim=s.embedding_dim,
            base_url=s.base_url,
        )
    )

    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            api_key=s.api_key,
            model=s.graphiti_llm_model,
            small_model=s.graphiti_llm_model,
            base_url=s.base_url,
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

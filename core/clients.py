"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pydantic import BaseModel, ConfigDict

from cogram.cross_encoder import CrossEncoderClient
from cogram.driver.driver import GraphDriver
from cogram.embedder import EmbedderClient
from cogram.llm_client import LLMClient
from cogram.utils.tracer import Tracer


class CogramClients(BaseModel):
    """Bundle of clients passed to Cogram's internal modules.

    Wraps the upstream graphiti GraphitiClients shape; exported under both
    names for backward compatibility with anything importing the old name.
    """

    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
    tracer: Tracer

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Backward-compat alias (graphiti's original name)
GraphitiClients = CogramClients

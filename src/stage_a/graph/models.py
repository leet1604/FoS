from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    node_type: str
    attributes: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    attributes: dict = Field(default_factory=dict)
    provenance_ids: list[str] = Field(default_factory=list)


class SerializableGraph(BaseModel):
    metadata: dict = Field(default_factory=dict)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Детерминированное разбиение Лувена с учётом суммарного веса связей."""
    if not graph:
        return []
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes(data=True)))
    for source, target, data in sorted(graph.edges(data=True)):
        weight = float(data.get("weight", 1))
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes[0])
    return [Community(community_id=index, nodes=nodes) for index, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Считает статистику программно и просит LLM только изложить факты."""
    nodes = sorted(set(community.nodes) & set(graph.nodes))
    subgraph = graph.subgraph(nodes)
    types = Counter(graph.nodes[node].get("type", node.split(":", 1)[0]) for node in nodes)
    relations = Counter()
    lines = []
    for source, target, data in sorted(subgraph.edges(data=True)):
        kind = data.get("type", "UNKNOWN")
        weight = data.get("weight", 1)
        relations[kind] += weight
        lines.append(f"{source} --{kind}--> {target}; вес={weight}")
    prompt = (
        f"Сообщество {community.community_id}. Сущности ({len(nodes)}): {', '.join(nodes)}.\n"
        f"Число связей: {subgraph.number_of_edges()}. Типы сущностей: {dict(sorted(types.items()))}.\n"
        f"Частоты связей (сумма весов): {dict(sorted(relations.items()))}.\n"
        "Факты по связям:\n" + "\n".join(lines) + "\n"
        "Напиши на русском языке сводку до 150 слов. Не придумывай факты и числа; "
        "выдели сущности, связи и наиболее частые события."
    )
    result = llm.chat([{"role": "user", "content": prompt}], max_tokens=450).content.strip()
    return " ".join(result.split()[:150])

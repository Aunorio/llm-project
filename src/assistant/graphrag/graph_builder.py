"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    graph = nx.DiGraph()

    for result in results:
        chunk_id = result.chunk_id

        for entity in result.entities:
            node_id = entity.node_id

            if node_id not in graph:
                graph.add_node(
                    node_id,
                    type=entity.type,
                    name=entity.name,
                    sources=[],
                )

            sources = graph.nodes[node_id]["sources"]

            if chunk_id not in sources:
                sources.append(chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            subject_id = relation.subject.node_id
            object_id = relation.object.node_id

            if not graph.has_node(subject_id):
                continue

            if not graph.has_node(object_id):
                continue

            if graph.has_edge(subject_id, object_id):
                edge = graph.edges[subject_id, object_id]

                if edge["type"] != relation.type:
                    continue
            else:
                graph.add_edge(
                    subject_id,
                    object_id,
                    type=relation.type,
                    weight=0,
                    sources=[],
                )

            edge = graph.edges[subject_id, object_id]

            if chunk_id not in edge["sources"]:
                edge["sources"].append(chunk_id)
                edge["weight"] = len(edge["sources"])

    conflicts = 0
    edges_to_remove = set()
    checked_pairs = set()

    for subject_id, object_id, data in list(graph.edges(data=True)):
        if data["type"] != "REPLACED_BY":
            continue

        reverse_edge = graph.get_edge_data(object_id, subject_id)

        if reverse_edge is None or reverse_edge["type"] != "REPLACED_BY":
            continue

        pair = frozenset((subject_id, object_id))

        if pair in checked_pairs:
            continue

        checked_pairs.add(pair)
        conflicts += 1

        forward_weight = data["weight"]
        reverse_weight = reverse_edge["weight"]

        if forward_weight > reverse_weight:
            edges_to_remove.add((object_id, subject_id))
        elif reverse_weight > forward_weight:
            edges_to_remove.add((subject_id, object_id))
        else:
            edges_to_remove.add((subject_id, object_id))
            edges_to_remove.add((object_id, subject_id))

    for subject_id, object_id in edges_to_remove:
        if graph.has_edge(subject_id, object_id):
            graph.remove_edge(subject_id, object_id)

    graph.graph["conflicts"] = conflicts

    return graph

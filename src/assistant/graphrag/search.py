"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def _documents(sources) -> list[str]:
    return sorted({str(source).split("#", 1)[0] for source in sources})


def _replacement_edges(graph: nx.DiGraph, nodes: set[str]) -> set[tuple[str, str]]:
    """Дойти по направленным цепочкам замен от найденных деталей."""
    edges = set()
    visited = set()
    pending = sorted(node for node in nodes if node.startswith("Part:"))
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        for successor in graph.successors(node):
            if graph[node][successor].get("type") == "REPLACED_BY":
                edges.add((node, successor))
                pending.append(successor)
    return edges


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отбирает только факты на коротких путях между опорами или в окрестности опоры."""
    graph = kb.graph
    codes = find_codes(question)
    anchors = {f"{code_type(code)}:{code}" for code in codes}
    if any(node not in graph for node in anchors):
        return Answer("Не найден указанный код в базе знаний.", refused=True)
    anchors.update(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary) if f"{kind}:{name}" in graph)
    if not anchors:
        return Answer("Не удалось определить сущности вопроса.", refused=True)

    undirected = graph.to_undirected()
    selected: set[tuple[str, str]] = set()
    if len(anchors) >= 2:
        for index, source in enumerate(sorted(anchors)):
            for target in sorted(anchors)[index + 1 :]:
                for path in nx.all_simple_paths(undirected, source, target, cutoff=3):
                    for first, second in zip(path, path[1:]):
                        if graph.has_edge(first, second):
                            selected.add((first, second))
                        if graph.has_edge(second, first):
                            selected.add((second, first))
    else:
        root = next(iter(anchors))
        distances = nx.single_source_shortest_path_length(undirected, root, cutoff=2)
        selected = {(a, b) for a, b in graph.edges if a in distances and b in distances
                    and min(distances[a], distances[b]) < 2}

    path_nodes = set(anchors)
    for first, second in selected:
        path_nodes.update((first, second))
    selected.update(_replacement_edges(graph, path_nodes))
    if not selected:
        return Answer("В графе нет фактов, достаточных для ответа.", refused=True)

    facts = []
    sources = set()
    for number, (first, second) in enumerate(sorted(selected), start=1):
        data = graph[first][second]
        documents = _documents(data.get("sources", []))
        sources.update(documents)
        facts.append(f"[{number}] {first} --{data.get('type', 'RELATED')}--> {second}; "
                     f"вес={data.get('weight', 1)}; источники: {', '.join(documents) or 'не указаны'}")
    prompt = ("Ответь на вопрос только по пронумерованным фактам ниже. Не добавляй сведений "
              "извне. Если фактов недостаточно, прямо сообщи об этом.\n"
              f"Вопрос: {question}\nФакты:\n" + "\n".join(facts))
    response = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    return Answer(text=response, sources=sorted(sources), facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Map по всем непустым сводкам; Reduce только для полезных JSON-ответов."""
    partials = []
    for community in kb.communities:
        if not community.summary.strip():
            continue
        prompt = (f"Вопрос: {question}\nСводка сообщества {community.community_id}: "
                  f"{community.summary}\nОтветь строго JSON: answer — частичный ответ только по сводке, "
                  "score — целое число полезности от 0 до 10. При отсутствии фактов score=0.")
        response = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA)
        try:
            data = json.loads(response.content)
            score = data.get("score")
            answer = data.get("answer")
            if type(score) is int and 0 < score <= 10 and isinstance(answer, str) and answer.strip():
                partials.append((score, community.community_id, answer.strip()))
        except (ValueError, TypeError, AttributeError):
            continue
    if not partials:
        return Answer("В сводках сообществ нет достаточных сведений для ответа.", refused=True)
    partials.sort(key=lambda item: (-item[0], item[1]))
    facts = [f"Сообщество {community_id} (полезность {score}/10): {answer}"
             for score, community_id, answer in partials]
    prompt = (f"Вопрос: {question}\nОбъедини частичные ответы в один точный ответ, "
              "без выдуманных сведений и повторов:\n" + "\n".join(facts))
    response = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    return Answer(text=response, facts=facts)
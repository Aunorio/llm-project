"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM_PROMPT = """
Ты извлекаешь сущности и связи из документов сервисной службы.

Используй исключительно информацию, которая прямо подтверждается текстом.

Допустимые типы сущностей:
- ProductLine — линейка оборудования;
- Model — модель оборудования;
- Component — узел оборудования;
- Part — номер детали;
- ErrorCode — код ошибки;
- Procedure — номер регламента;
- Supplier — поставщик.

Допустимые связи и их направление:

- BELONGS_TO: Model -> ProductLine.
  Модель входит в линейку.

- HAS_PART: Model -> Part.
  В модели установлена деталь.

- IS_A: Part -> Component.
  Деталь является узлом данного вида.

- INDICATES: ErrorCode -> Component.
  Код ошибки указывает на отказ узла.

- FIXED_BY: ErrorCode -> Procedure.
  Ошибка устраняется по регламенту.

- REPLACED_BY: Part -> Part.
  Старая деталь заменена новой.

- SUPPLIED_BY: Part -> Supplier.
  Деталь поставляется поставщиком.

- FAILED: Model -> Component.
  В модели зафиксирован отказ узла.

Не меняй направление связей.
Не придумывай отсутствующие сущности, коды и связи.
Коды переписывай в каноническом виде.
Названия узлов, линеек и поставщиков используй в именительном падеже.

Верни только JSON с двумя массивами: entities и relations.

Пример:

Текст:
"Модель AR-210 входит в линейку Арктика.
В модели установлена деталь P-1042."

Ответ:
{
  "entities": [
    {"type": "Model", "name": "AR-210"},
    {"type": "ProductLine", "name": "Арктика"},
    {"type": "Part", "name": "P-1042"}
  ],
  "relations": [
    {
      "subject_type": "Model",
      "subject": "AR-210",
      "type": "BELONGS_TO",
      "object_type": "ProductLine",
      "object": "Арктика"
    },
    {
      "subject_type": "Model",
      "subject": "AR-210",
      "type": "HAS_PART",
      "object_type": "Part",
      "object": "P-1042"
    }
  ]
}
"""


def empty_result(chunk: Chunk) -> ExtractionResult:
    """Создаёт пустой результат, сохраняя идентификаторы чанка."""
    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
    )

def parse_response(content: str) -> dict | None:
    """Проверяет, что ответ является JSON нужной структуры."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    entities = data.get("entities")
    relations = data.get("relations")

    if not isinstance(entities, list) or not isinstance(relations, list):
        return None

    for item in entities:
        if not isinstance(item, dict):
            return None

        if not isinstance(item.get("type"), str):
            return None

        if not isinstance(item.get("name"), str):
            return None

    required_relation_fields = (
        "subject_type",
        "subject",
        "type",
        "object_type",
        "object",
    )

    for item in relations:
        if not isinstance(item, dict):
            return None

        if any(not isinstance(item.get(field), str) for field in required_relation_fields):
            return None

    return data

def ground_entity(
    entity_type: str,
    raw_name: str,
    text: str,
    codes_in_text: set[str],
    words: dict[str, list[str]],
) -> Entity | None:
    """Проверяет сущность по исходному тексту и возвращает её канонический вариант."""

    if entity_type not in ENTITY_TYPES:
        return None

    canonical = canonical_name(entity_type, raw_name, words)

    if canonical is None:
        return None

    if entity_type in ("Model", "Part", "ErrorCode", "Procedure"):
        if canonical not in codes_in_text:
            return None
    else:
        if canonical.casefold() not in text.casefold():
            return None

    return Entity(entity_type, canonical)

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    words = vocabulary()
    codes_in_text = set(find_codes(chunk.text))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Извлеки сущности и связи из следующего чанка.\n\n"
                f"ID документа: {chunk.doc_id}\n"
                f"ID чанка: {chunk.chunk_id}\n\n"
                f"Текст чанка:\n{chunk.text}"
            ),
        },
    ]

    data = None

    for _ in range(2):
        try:
            response = llm.chat(
                messages,
                json_schema=EXTRACTION_SCHEMA,
            )
            data = parse_response(response.content)

            if data is not None:
                break

        except Exception:
            data = None

    if data is None:
        return empty_result(chunk)

    result = empty_result(chunk)

    valid_entities: dict[str, Entity] = {}

    for item in data["entities"]:
        entity = ground_entity(
            item["type"],
            item["name"],
            chunk.text,
            codes_in_text,
            words,
        )

        if entity is None:
            result.dropped += 1
            continue

        valid_entities[entity.node_id] = entity

    for item in data["relations"]:
        subject_type = item["subject_type"]
        relation_type = item["type"]
        object_type = item["object_type"]

        if RELATION_SIGNATURES.get(relation_type) != (
            subject_type,
            object_type,
        ):
            result.dropped += 1
            continue

        subject = ground_entity(
            subject_type,
            item["subject"],
            chunk.text,
            codes_in_text,
            words,
        )

        obj = ground_entity(
            object_type,
            item["object"],
            chunk.text,
            codes_in_text,
            words,
        )

        if subject is None or obj is None:
            result.dropped += 1
            continue

        valid_entities[subject.node_id] = subject
        valid_entities[obj.node_id] = obj

        result.relations.append(
            Relation(
                subject=subject,
                type=relation_type,
                object=obj,
            )
        )

    result.entities = list(valid_entities.values())

    return result

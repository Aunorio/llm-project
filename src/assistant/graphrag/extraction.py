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

MAX_TEXT_LENGTH = 500

SYSTEM_PROMPT = """
Ты извлекаешь сущности и связи из документов сервисной службы.

Используй исключительно информацию, которая прямо подтверждается текстом.

Допустимые типы сущностей:
- ProductLine - линейка оборудования;
- Model - модель оборудования;
- Component - узел оборудования;
- Part - номер детали;
- ErrorCode - код ошибки;
- Procedure - номер регламента;
- Supplier - поставщик.

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

ВАЖНЫЕ ПРАВИЛА ИЗВЛЕЧЕНИЯ:

1. Если в тексте указано, что в модели установлен определённый узел,
   а рядом записан номер детали, создавай HAS_PART:
   Model -> Part.

2. Если строка имеет вид:
   "компрессор - деталь P-1010",
   то "компрессор" - это Component,
   а "P-1010" - это Part.
   Создавай IS_A:
   Part -> Component.

3. Если в той же строке указан поставщик,
   создавай SUPPLIED_BY:
   Part -> Supplier.

4. Если во фрагменте нет повторного упоминания модели,
   но в переданном контексте указано, что перечисленные узлы
   установлены в определённой модели, используй эту модель
   для создания HAS_PART.

5. Для каждой строки состава узлов извлекай все связи,
   которые подтверждаются этой строкой и контекстом:
   HAS_PART, IS_A и SUPPLIED_BY.

6. Внимательно проверяй направление каждой связи.
   HAS_PART - Model -> Part.
   IS_A - Part -> Component.
   SUPPLIED_BY - Part -> Supplier.

Пример:

Текст:
"Модель AR-210.
В AR-210 установлены следующие узлы:
пусковое реле - деталь P-1042,
поставщик ХолодКомплект."

Ответ:
{
  "entities": [
    {"type": "Model", "name": "AR-210"},
    {"type": "Component", "name": "пусковое реле"},
    {"type": "Part", "name": "P-1042"},
    {"type": "Supplier", "name": "ХолодКомплект"}
  ],
  "relations": [
    {
      "subject_type": "Model",
      "subject": "AR-210",
      "type": "HAS_PART",
      "object_type": "Part",
      "object": "P-1042"
    },
    {
      "subject_type": "Part",
      "subject": "P-1042",
      "type": "IS_A",
      "object_type": "Component",
      "object": "пусковое реле"
    },
    {
      "subject_type": "Part",
      "subject": "P-1042",
      "type": "SUPPLIED_BY",
      "object_type": "Supplier",
      "object": "ХолодКомплект"
    }
  ]
}

Другой пример:

Текст:
"Деталь P-1042 заменена на P-1042B."

Ответ:
{
  "entities": [
    {"type": "Part", "name": "P-1042"},
    {"type": "Part", "name": "P-1042B"}
  ],
  "relations": [
    {
      "subject_type": "Part",
      "subject": "P-1042",
      "type": "REPLACED_BY",
      "object_type": "Part",
      "object": "P-1042B"
    }
  ]
}

Верни только JSON с двумя массивами: entities и relations.
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

        if any(
            not isinstance(item.get(field), str)
            for field in required_relation_fields
        ):
            return None

    return data

def get_context_lines(lines: list[str]) -> list[str]:
    """
    Выбирает строки, которые нужно повторять в каждом фрагменте.
    Сохраняет заголовок документа, описание состава узлов
    и предложение, связывающее узлы с моделью.
    """
    context = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        lowered = stripped.casefold()

        if stripped.startswith("#"):
            context.append(line)
            continue

        if "состав узлов" in lowered:
            context.append(line)
            continue

        if any(
            word in lowered
            for word in (
                "установлен",
                "установлены",
                "установлена",
                "установлено",
            )
        ):
            context.append(line)

    if not context:
        for line in lines:
            if line.strip():
                context.append(line)
                break

    return context

def split_text(text: str) -> list[str]:
    """Делит чанк на фрагменты с сохранением контекста."""
    if len(text) <= MAX_TEXT_LENGTH:
        return [text]

    lines = text.splitlines()

    if len(lines) <= 1:
        return [
            text[i:i + MAX_TEXT_LENGTH]
            for i in range(0, len(text), MAX_TEXT_LENGTH)
        ]

    context = get_context_lines(lines)
    context_set = set(context)

    body = [
        line
        for line in lines
        if line not in context_set and line.strip()
    ]

    context_text = "\n".join(context)
    context_length = len(context_text) + 1

    parts = []
    current_lines = []
    current_length = context_length

    for line in body:
        line_length = len(line) + 1

        if (
            current_lines
            and current_length + line_length > MAX_TEXT_LENGTH
        ):
            parts.append(
                context_text + "\n" + "\n".join(current_lines)
            )

            current_lines = []
            current_length = context_length

        current_lines.append(line)
        current_length += line_length

    if current_lines:
        parts.append(
            context_text + "\n" + "\n".join(current_lines)
        )

    return parts or [text]

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

def normalize_relation(item: dict) -> dict:
    """
    Исправляет направление связи, если модель вернула
    субъект и объект в обратном порядке.
    Допустимое направление определяется RELATION_SIGNATURES.
    """
    relation = dict(item)

    relation_type = relation["type"]
    expected = RELATION_SIGNATURES.get(relation_type)

    if expected is None:
        return relation

    actual = (
        relation["subject_type"],
        relation["object_type"],
    )

    if actual == expected[::-1] and actual != expected:
        relation["subject_type"], relation["object_type"] = (
            relation["object_type"],
            relation["subject_type"],
        )

        relation["subject"], relation["object"] = (
            relation["object"],
            relation["subject"],
        )

    return relation

def request_extraction(
    text: str,
    chunk: Chunk,
    llm: LLMClient,
) -> dict | None:
    """Запрашивает извлечение с одной повторной попыткой."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Извлеки сущности и связи из следующего фрагмента чанка.\n\n"
                "Контекстные строки могут содержать сведения о модели "
                "и составе её узлов. Используй этот контекст при "
                "извлечении связей HAS_PART.\n\n"
                f"ID документа: {chunk.doc_id}\n"
                f"ID чанка: {chunk.chunk_id}\n\n"
                f"Текст фрагмента:\n{text}"
            ),
        },
    ]

    for _ in range(2):
        try:
            response = llm.chat(
                messages,
                json_schema=EXTRACTION_SCHEMA,
                max_tokens=4096,
            )
            data = parse_response(response.content)

            if data is not None:
                return data

        except Exception:
            pass

    return None

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлекает сущности и связи из чанка, объединяя результаты фрагментов."""
    words = vocabulary()

    codes_in_text = set(find_codes(chunk.text))

    text_parts = split_text(chunk.text)

    result = empty_result(chunk)

    valid_entities: dict[str, Entity] = {}
    valid_relations: dict[tuple[str, str, str], Relation] = {}

    for part_text in text_parts:
        data = request_extraction(part_text, chunk, llm)

        if data is None:
            return empty_result(chunk)

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

        for raw_relation in data["relations"]:
            item = normalize_relation(raw_relation)

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

            relation = Relation(
                subject=subject,
                type=relation_type,
                object=obj,
            )

            relation_key = (
                subject.node_id,
                relation_type,
                obj.node_id,
            )

            valid_relations[relation_key] = relation

    result.entities = list(valid_entities.values())
    result.relations = list(valid_relations.values())

    return result

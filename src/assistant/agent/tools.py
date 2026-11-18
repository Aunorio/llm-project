"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Отвечает на вопрос о моделях, кодах ошибок, деталях, узлах и регламентах "
                "по базе знаний (графу знаний). Возвращает ответ, факты, источники и признак отказа."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "вопрос по документам"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Отвечает на обзорный вопрос по всему корпусу документов: сводит ответы "
                "по тематическим сообществам графа. Возвращает ответ, источники и признак отказа."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "обзорный вопрос"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Возвращает остаток детали на складе и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "номер детали вида P-1042 или P-1042B"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Возвращает цену детали в рублях и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "номер детали вида P-1042 или P-1042B"}
                },
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": (
                "Создаёт заказ детали на склад. Количество должно быть названо пользователем. "
                "Возвращает номер созданного заказа."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "номер детали вида P-1042 или P-1042B"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string", "description": "код модели вида AR-210"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    raise NotImplementedError("Реализуйте build_tools")

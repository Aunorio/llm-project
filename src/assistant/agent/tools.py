"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
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
                "properties": {"part_id": {"type": "string", "description": "номер детали вида P-1042 или P-1042B"}},
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
    def validate_part_id(part_id: str) -> tuple[str | None, dict | None]:
        if not isinstance(part_id, str):
            return None, {"error": "номер детали должен быть строкой"}

        code = normalize_code(part_id)
        if code is None or code_type(code) != "Part":
            return None, {"error": f"номер детали должен быть вида P-1042 или P-1042B, получено: {part_id!r}"}

        return code, None

    def validate_warehouse(warehouse: str) -> dict | None:
        if warehouse not in WAREHOUSES:
            return {"error": f"неизвестный склад {warehouse!r}: выберите один из {', '.join(WAREHOUSES)}"}
        return None

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {
            "answer": answer.text,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def check_stock(part_id: str, warehouse: str) -> dict:
        code, error = validate_part_id(part_id)
        if error:
            return error

        error = validate_warehouse(warehouse)
        if error:
            return error

        try:
            return erp.get_stock(code, warehouse)
        except ErpNotFound:
            return {"error": f"деталь {code} в ERP не найдена: данных нет"}
        except ErpRejected as exc:
            return {"error": f"запрос отклонён ERP: {exc}"}

    def get_price(part_id: str) -> dict:
        code, error = validate_part_id(part_id)
        if error:
            return error

        try:
            return erp.get_price(code)
        except ErpNotFound:
            return {"error": f"деталь {code} в ERP не найдена: данных нет"}
        except ErpRejected as exc:
            return {"error": f"запрос отклонён ERP: {exc}"}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        code, error = validate_part_id(part_id)
        if error:
            return error

        error = validate_warehouse(warehouse)
        if error:
            return error

        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": f"количество должно быть целым числом больше нуля, получено: {qty!r}"}

        if not isinstance(model, str) or not model.strip():
            return {"error": "необходимо указать модель"}

        try:
            return erp.create_order(code, warehouse, qty, model)
        except ErpNotFound:
            return {"error": f"деталь {code} в ERP не найдена: данных нет"}
        except ErpRejected as exc:
            return {"error": f"заказ отклонён ERP: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }

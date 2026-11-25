"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

KNOWLEDGE_TOOLS = ("search_knowledge", "summarize_topic")
ORDER_TOOL = "create_service_order"
ORDER_FIELDS = ("part_id", "warehouse", "qty", "model")
MAX_ERP_ATTEMPTS = 3

REFUSAL_TEXT = "В базе знаний не найдено: ответ не подтверждён данными инструментов."

# Число с разрядами через пробел, за которым стоит единица измерения —
# так оценщик в infra/evaluation.py отличает цены и остатки от прочих чисел.
_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)")

SYSTEM_PROMPT = """Ты — ассистент диспетчера сервисной службы «ПолярТех».
По модели и коду ошибки помогаешь выбрать деталь, проверить остаток и цену в ERP
и подготовить заказ.
- Сначала ищи ответ в базе знаний через search_knowledge; обзорные вопросы задавай summarize_topic.
- Если деталь снята с производства, проверь в результатах search_knowledge, есть ли связь REPLACED_BY.
- Если замена найдена, используй номер новой детали для проверки
  остатка через check_stock и цены через get_price.
- Не рекомендуй снятую с производства деталь, если для неё найдена замена.
- В итоговом ответе укажи исходную деталь и её актуальную замену.
- Номера деталей для ERP бери только из вопроса или результатов инструментов.
- Остаток узнавай через check_stock, цену — через get_price.
- Заказ оформляет create_service_order; количество должно быть названо пользователем.
- Если данных нет, честно скажи, что в базе знаний не найдено.
- Отвечай только фактами из результатов инструментов и называй источники.
"""


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: list[str] = []
    seen_codes = set(find_codes(question))
    known_text = question
    cache: dict[tuple[str, str], dict] = {}
    kb_refused = False

    def finish(answer: str, *, refused: bool = False) -> AgentResult:
        return AgentResult(
            answer=answer,
            steps=steps,
            sources=sorted(set(sources)),
            refused=refused or kb_refused or no_data(),
        )

    def no_data() -> bool:
        """True, если ни один инструмент не вернул подтверждённых данных."""
        if not steps:
            return False

        for step in steps:
            if step.error is not None or not step.result or "error" in step.result:
                continue
            if step.tool in KNOWLEDGE_TOOLS and step.result.get("refused"):
                continue
            return False

        return True

    def check_answer(answer: str) -> str | None:
        """Проверяет, что коды и числовые значения ответа подтверждены данными."""
        for code in find_codes(answer):
            if code not in seen_codes:
                return f"код {code}"

        known_numbers = {_digits(raw) for raw in re.findall(_NUMBER, known_text)}

        for raw, unit in _WITH_UNIT.findall(answer):
            if _digits(raw) not in known_numbers:
                return f"число {_digits(raw)} {unit}"

        return None

    def execute(call) -> dict | AgentResult:
        """Проверяет вызов инструмента и выполняет его."""
        name, args = call.name, call.arguments

        if "_raw" in args:
            return {"error": "аргументы вызова не читаются: ожидался JSON-объект"}

        func = tools.get(name)
        if func is None:
            return {"error": f"неизвестный инструмент: {name}"}

        part_id = args.get("part_id")
        if part_id is not None:
            codes = find_codes(str(part_id))
            if not codes or any(code not in seen_codes for code in codes):
                return {
                    "error": (
                        f"номер детали {part_id} не встречается ни в вопросе, ни в результатах "
                        "инструментов. Сначала вызовите search_knowledge."
                    )
                }

        if name == ORDER_TOOL:
            qty = args.get("qty")
            qty_named = (
                isinstance(qty, int) and not isinstance(qty, bool) and bool(re.search(rf"(?<!\d){qty}(?!\d)", question))
            )

            if not qty_named:
                return AgentResult(answer="Уточните, пожалуйста, сколько штук заказать.")

            if not confirm:
                clean = {field: args[field] for field in ORDER_FIELDS if field in args}
                return AgentResult(
                    answer="Заказ готов к оформлению, жду подтверждения.",
                    needs_confirmation=True,
                    pending_action={"tool": ORDER_TOOL, "args": clean},
                )

        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in cache:
            return cache[key]

        attempts = 0

        while True:
            try:
                result = func(**args)
                break
            except ErpUnavailable:
                attempts += 1
                if attempts >= MAX_ERP_ATTEMPTS:
                    result = {"error": "данные ERP получить не удалось: сервис недоступен"}
                    break

        cache[key] = result
        return result

    def process_calls(response) -> AgentResult | None:
        """Выполняет вызовы инструментов из ответа модели."""
        nonlocal kb_refused, known_text

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                return None

            step = ToolStep(
                tool=call.name,
                args=dict(call.arguments),
            )
            steps.append(step)

            outcome = execute(call)

            if isinstance(outcome, AgentResult):
                outcome.steps = steps
                outcome.sources = sorted(set(sources))
                return outcome

            messages.append(tool_message(call, outcome))

            blob = json.dumps(outcome, ensure_ascii=False)
            known_text += "\n" + blob
            seen_codes.update(find_codes(blob))

            step.result = outcome

            if "error" in outcome:
                step.error = outcome["error"]

            if call.name in KNOWLEDGE_TOOLS:
                sources.extend(outcome.get("sources") or [])
                if outcome.get("refused"):
                    kb_refused = True

        return None

    def get_final_answer() -> AgentResult:
        """Получает и проверяет финальный ответ модели."""
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if response.tool_calls:
            return finish(REFUSAL_TEXT, refused=True)

        problem = check_answer(response.content)

        if problem is None:
            return finish(response.content)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Ответ не подтверждён данными ({problem}). "
                    "Исправь его, опираясь только на результаты инструментов. "
                    "Не добавляй неподтверждённые коды, цены и остатки."
                ),
            }
        )

        corrected = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(corrected.as_message())

        if corrected.tool_calls:
            return finish(REFUSAL_TEXT, refused=True)

        if check_answer(corrected.content) is None:
            return finish(corrected.content)

        return finish(REFUSAL_TEXT, refused=True)

    while True:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            problem = check_answer(response.content)

            if problem is None:
                return finish(response.content)

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Ответ не подтверждён данными ({problem}). "
                        "Исправь его, опираясь только на результаты инструментов."
                    ),
                }
            )

            corrected = llm.chat(messages, tools=TOOL_SCHEMAS)
            messages.append(corrected.as_message())

            if corrected.tool_calls:
                early = process_calls(corrected)
                if early is not None:
                    return early
                continue

            if check_answer(corrected.content) is None:
                return finish(corrected.content)

            return finish(REFUSAL_TEXT, refused=True)

        early = process_calls(response)
        if early is not None:
            return early

        if len(steps) >= max_steps:
            return get_final_answer()

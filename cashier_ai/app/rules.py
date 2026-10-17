"""Каталог типов нарушений и привязка к нормативке.

ВАЖНО (критерий 4: «нет выдуманных ссылок»): поля document/item заполняйте
ТОЛЬКО после того, как вручную открыли акт на adilet.zan.kz / nationalbank.kz
и нашли точный пункт. Пока verified=False, в ответ идёт пометка, а не ссылка.
"""

# cluster_id (из artifacts/clusters/*) -> ключ типа. Заполняется вручную
# после просмотра примеров кадров по каждому кластеру.
CLUSTER_TO_TYPE = {
    0: "unknown",
    1: "unattended_cash",
    2: "cash_acceptance_without_client_identification",
    3: "",
    4: ""
}

RULES = {
    "unattended_cash": {
        "title": "Наличные без присмотра / вне контроля",          # пример, замените
        "what": "денежные средства оставлены на стойке без контроля кассира",
        "document": "",   # TODO: точное название акта
        "item": "",       # TODO: пункт/раздел
        "verified": False,
    },
    "unauthorized_person_in_zone": {
        "title": "Посторонний в кассовой зоне",                   # пример, замените
        "what": "в зоне кассира находится человек, не участвующий в операции",
        "document": "",
        "item": "",
        "verified": False,
    },
    "unknown": {
        "title": "Подозрительное событие (тип не определён)",
        "what": "модель обнаружила отклонение от типичной картины корректной работы",
        "document": "",
        "item": "",
        "verified": False,
    },
    "cash_acceptance_without_client_identification": {
    "title": "Приём наличных без идентификации клиента",
    "what": "на видео не видно проверки личности клиента перед приёмом наличных",
    "document": "Закон Республики Казахстан от 26 июля 2016 года № 11-VI ЗРК «О платежах и платежных системах»",
    "item": "статья 13, пункт 12, подпункт 1",
    "verified": False,
},
}


def rule_for(type_key):
    r = RULES.get(type_key, RULES["unknown"])
    if r["verified"] and r["document"]:
        src = {"document": r["document"], "item": r["item"]}
    else:
        src = {"document": None, "item": None,
               "note": "Нормативный источник требует ручного подтверждения"}
    return r, src

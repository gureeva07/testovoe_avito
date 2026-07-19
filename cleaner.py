import re
import html
from bs4 import BeautifulSoup

# эти слова в начале запроса ничего не значат для поиска, убираю их
GREETINGS = re.compile(
    r"(здравствуйте|добрый день|добрый вечер|доброе утро|привет|до свидания|всего доброго|"
    r"подскажите пожалуйста|подскажите|пожалуйста|помогите|"
    r"скажите пожалуйста|можете подсказать|помогите пожалуйста|"
    r"прошу помочь|заранее спасибо|спасибо)[,!. ]*",
    re.IGNORECASE,
)

# предлоги и союзы, смысла не несут, вопросительные слова (как, где, почему) специально оставила
STOP_WORDS = {
    "и", "а", "но", "или", "не", "в", "на", "с", "по", "за", "к", "у",
    "о", "об", "из", "до", "для", "от", "при", "про", "без", "под",
    "то", "это", "же", "бы", "ли", "уже", "ещё", "так", "вот",
}


def clean_query(query: str) -> str:
    text = GREETINGS.sub(" ", query)
    # убираю html-теги если вдруг попали в запрос
    text = re.sub(r"<[^>]+>", " ", text)
    words = [w for w in text.split() if w.lower() not in STOP_WORDS]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_html(raw_html: str) -> str:
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # без этого пункты списка слипаются в одну строку без пробелов
    for tag in soup.find_all(["li", "tr", "p", "br", "div"]):
        tag.append(" | ")

    text = soup.get_text(separator=" ")

    # &nbsp; и прочее превращаю в нормальные символы
    text = html.unescape(text)

    # ссылки для поиска бесполезны
    text = re.sub(r"https?://\S+", " ", text)

    # эмодзи и иконки убираю, они шумят
    text = re.sub(r"[^\u0000-\uFFFF]", " ", text)
    text = re.sub(r"[\u2000-\u2BFF\u2E00-\u2E7F]", " ", text)

    text = re.sub(r"\s+", " ", text)

    # заменяю временный разделитель на точку
    text = text.replace(" | ", ". ")
    text = text.replace(" |", ".")
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()

import os
import json
import logging
import hashlib
import requests
import time
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import markdown

# =============================================================================
#  CONFIGURATION
#
#  Scope: Asia-Pacific EXCLUDING Mainland China (covered by china_eco_agent.py
#  and China_tax_law separately). Includes Hong Kong and Taiwan (not part of
#  the Mainland China report), plus Oceania (Australia, New Zealand, Pacific
#  island nations).
#
#  IMPORTANT — division of labour with sibling agents:
#  Aviation / GSE demand signals (airports, airlines, passenger & cargo
#  traffic, fleet orders) are OUT OF SCOPE here — they are already covered
#  by apac_gse_agent.py. This agent focuses on general macroeconomic and
#  competitiveness signals: manufacturing PMI, FDI & supply-chain
#  diversification ("China+1"), FX, monetary policy & financing conditions,
#  trade/tariffs, structural reform, and regulatory compliance.
# =============================================================================
load_dotenv()
LOG_FILE  = Path("logs/agent_apac_eco.log")
SEEN_FILE = Path("seen_apac_eco_articles.json")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Max articles to enrich with body excerpt
ENRICH_MAX = 15
# Max articles sent to DeepSeek in one call
DEEPSEEK_MAX_ARTICLES = 50

# =============================================================================
#  COUNTRY TIERS — used to weight Tavily query allocation and, indirectly,
#  which signals get surfaced. Tier is informational (logged / included in
#  the DeepSeek prompt); filtering itself remains keyword-driven.
# =============================================================================
COUNTRIES_PRIMARY = [
    "Japan", "South Korea", "India", "Indonesia", "Thailand",
    "Vietnam", "Philippines", "Hong Kong", "Taiwan",
]

COUNTRIES_SECONDARY = [
    "Malaysia", "Singapore", "Myanmar", "Bangladesh", "Nepal",
    "Cambodia", "Laos", "Bhutan", "Sri Lanka", "Pakistan",
]

COUNTRIES_OCEANIA = [
    "Australia", "New Zealand", "Fiji", "Papua New Guinea",
    "Solomon Islands", "Samoa", "Vanuatu", "Tonga", "Kiribati",
]

ALL_COUNTRIES = COUNTRIES_PRIMARY + COUNTRIES_SECONDARY + COUNTRIES_OCEANIA

# =============================================================================
#  ALERT THRESHOLDS — used in system prompt and signal classification
# =============================================================================
THRESHOLDS = {
    "pmi_crisis":       48.0,   # PMI below this = contraction signal
    "pmi_boom":         52.0,   # PMI above this = expansion signal
    "fx_move_alert":     3.0,   # % move vs EUR or USD in 30 days
    "rate_move_alert":  50.0,   # bps policy rate move in one decision
    "fdi_swing_alert":  15.0,   # % YoY swing in FDI inflow/outflow
}

# =============================================================================
#  KEYWORDS — macro/competitiveness indicators relevant to a CFO overseeing
#  APAC operations. Deliberately EXCLUDES aviation/air-traffic/airport
#  terms (owned by apac_gse_agent.py) to avoid duplicate signal generation.
# =============================================================================
KEYWORDS_ECO_APAC = [
    # --- PMI / manufacturing health ---
    "manufacturing PMI", "PMI", "purchasing managers index",
    "factory output", "industrial production", "new export orders",
    "manufacturing output", "business confidence index",

    # --- GDP / growth ---
    "GDP growth", "GDP", "economic growth rate", "quarterly GDP",
    "growth forecast", "growth target", "recession", "economic slowdown",
    "economic recovery",

    # --- FDI / supply chain diversification ("China+1") ---
    "foreign direct investment", "FDI", "foreign investment inflow",
    "foreign investment outflow", "China plus one", "China+1",
    "supply chain diversification", "supply chain relocation",
    "nearshoring", "friendshoring", "manufacturing relocation",
    "factory relocation", "reshoring", "investment pledge",

    # --- FX / currency ---
    "currency", "exchange rate", "devaluation", "depreciation",
    "appreciation", "central bank intervention", "currency peg",
    "rupiah", "baht", "yen", "won", "rupee", "peso", "ringgit",
    "Vietnamese dong", "Taiwan dollar", "Hong Kong dollar",
    "Australian dollar", "New Zealand dollar",

    # --- Monetary policy / financing ---
    "interest rate", "policy rate", "rate hike", "rate cut",
    "central bank", "monetary policy", "monetary tightening",
    "monetary easing", "credit conditions", "bank lending",
    "sovereign bond yield", "credit rating",

    # --- Trade / tariffs ---
    "tariff", "trade war", "export control", "sanctions",
    "trade agreement", "RCEP", "CPTPP", "free trade agreement",
    "export growth", "import growth", "trade deficit", "trade surplus",
    "customs duty", "anti-dumping",

    # --- Labour / manufacturing costs ---
    "labour cost", "wage growth", "minimum wage", "labor shortage",
    "manufacturing cost", "unit labour cost",

    # --- Structural reform / industrial policy ---
    "structural reform", "economic reform", "privatization",
    "market liberalization", "industrial policy", "investment incentive",
    "special economic zone", "free trade zone", "tax incentive",

    # --- Regulatory / compliance risk ---
    "regulatory compliance", "data protection law", "cybersecurity law",
    "antitrust", "regulatory crackdown", "foreign ownership rule",
    "capital controls", "repatriation restriction",

    # --- Crisis / instability signals ---
    "debt crisis", "default", "bankruptcy", "political instability",
    "currency crisis", "capital flight", "credit downgrade",
    "sovereign default risk",
]

# =============================================================================
#  SCRAPED SOURCES — international / regional English-language outlets,
#  generally accessible from GitHub Actions US runners. Local-language and
#  harder-to-reach sources are handled via Tavily below.
# =============================================================================
SOURCES = [
    {
        "nom": "Nikkei Asia",
        "url": "https://asia.nikkei.com/Economy",
        "type": "scrape_generic",
        "selector": "article a, h3 a, .article-card a, a",
        "base_url": "https://asia.nikkei.com",
    },
    {
        "nom": "Reuters Asia Pacific",
        "url": "https://www.reuters.com/world/asia-pacific/",
        "type": "scrape_generic",
        "selector": "a[data-testid='Heading'], h3 a, article a",
        "base_url": "https://www.reuters.com",
    },
    {
        "nom": "Bloomberg Asia",
        "url": "https://www.bloomberg.com/asia",
        "type": "scrape_generic",
        "selector": "h3 a, .headline a, article h2 a",
        "base_url": "https://www.bloomberg.com",
    },
    {
        "nom": "Financial Times — Asia-Pacific",
        "url": "https://www.ft.com/asia-pacific",
        "type": "scrape_generic",
        "selector": "a[data-trackable='heading-link'], .o-teaser__heading a, h3 a, a",
        "base_url": "https://www.ft.com",
    },
    {
        "nom": "Channel News Asia — Business",
        "url": "https://www.channelnewsasia.com/business",
        "type": "scrape_generic",
        "selector": "h3 a, .h6__link, article a, a",
        "base_url": "https://www.channelnewsasia.com",
    },
    {
        "nom": "The Business Times (Singapore)",
        "url": "https://www.businesstimes.com.sg/international",
        "type": "scrape_generic",
        "selector": "h3 a, .article-title a, a",
        "base_url": "https://www.businesstimes.com.sg",
    },
    {
        "nom": "VnExpress International",
        "url": "https://e.vnexpress.net/news/business",
        "type": "scrape_generic",
        "selector": "h3 a, .title-news a, a",
        "base_url": "https://e.vnexpress.net",
    },
    {
        "nom": "Bangkok Post — Business",
        "url": "https://www.bangkokpost.com/business",
        "type": "scrape_generic",
        "selector": "h3 a, .article-info a, a",
        "base_url": "https://www.bangkokpost.com",
    },
    {
        "nom": "The Jakarta Post — Business",
        "url": "https://www.thejakartapost.com/business",
        "type": "scrape_generic",
        "selector": "h2 a, h3 a, .article-title a, a",
        "base_url": "https://www.thejakartapost.com",
    },
    {
        "nom": "Inquirer Business (Philippines)",
        "url": "https://business.inquirer.net/",
        "type": "scrape_generic",
        "selector": "h2 a, h3 a, .river_headline a, a",
        "base_url": "https://business.inquirer.net",
    },
    {
        "nom": "The Economic Times — India",
        "url": "https://economictimes.indiatimes.com/news/economy",
        "type": "scrape_generic",
        "selector": "h3 a, .eachStory a, a",
        "base_url": "https://economictimes.indiatimes.com",
    },
    {
        "nom": "The Korea Herald — Business",
        "url": "http://www.koreaherald.com/list.php?ct=020500000000",
        "type": "scrape_generic",
        "selector": "h4 a, .news_list a, a",
        "base_url": "http://www.koreaherald.com",
        "encoding": "utf-8",
    },
    {
        "nom": "Taipei Times — Business",
        "url": "https://www.taipeitimes.com/News/biz",
        "type": "scrape_generic",
        "selector": "h3 a, .archives a, a",
        "base_url": "https://www.taipeitimes.com",
    },
    {
        "nom": "South China Morning Post — Economy (HK)",
        "url": "https://www.scmp.com/economy",
        "type": "scrape_generic",
        "selector": "h2 a, h3 a, .article__title a, a",
        "base_url": "https://www.scmp.com",
    },
    {
        "nom": "Australian Financial Review — Economy",
        "url": "https://www.afr.com/economy",
        "type": "scrape_generic",
        "selector": "h3 a, .story-block__link, a",
        "base_url": "https://www.afr.com",
    },
    {
        "nom": "RNZ Pacific",
        "url": "https://www.rnz.co.nz/international/pacific-news",
        "type": "scrape_generic",
        "selector": "h3 a, .o-digest__title a, a",
        "base_url": "https://www.rnz.co.nz",
    },
    {
        "nom": "Trading Economics — Asia",
        "url": "https://tradingeconomics.com/matrix",
        "type": "scrape_generic",
        "selector": "table a, a",
        "base_url": "https://tradingeconomics.com",
    },
]

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def normaliser_url(url, base=None):
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    parsed = urlparse(url)
    url_propre = parsed._replace(query="", fragment="").geturl()
    if url_propre.endswith("/"):
        url_propre = url_propre[:-1]
    return url_propre


def charger_vus():
    if SEEN_FILE.exists():
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except Exception:
                return set()
    return set()


def sauvegarder_vus(vus):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(list(vus), f, ensure_ascii=False, indent=2)


def requeter_avec_retry(url, retries=3, timeout=20, **kwargs):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Attempt {i+1}/{retries} failed for {url}: {e}")
            time.sleep(2 ** i)
    return None


# =============================================================================
#  SCRAPERS
# =============================================================================

def scrape_generic(source):
    articles = []
    resp = requeter_avec_retry(source["url"])
    if not resp:
        return articles
    try:
        encoding = source.get("encoding", "utf-8")
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding=encoding)
        unique_links = {}
        for link in soup.select(source["selector"]):
            href  = link.get("href")
            titre = link.get_text(strip=True)
            if not href or not titre or len(titre) < 8:
                continue
            href = normaliser_url(href, source.get("base_url"))
            if href:
                unique_links[href] = titre
        for href, titre in list(unique_links.items())[:20]:
            articles.append({
                "source": source["nom"],
                "titre":  titre[:150],
                "lien":   href,
                "desc":   "",
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "id":     hashlib.md5((titre + href).encode()).hexdigest(),
            })
    except Exception as e:
        log.warning(f"Error scraping {source['nom']}: {e}")
    log.info(f"  Scraped {source['nom']}: {len(articles)} articles")
    return articles


def collecter_tous_articles():
    tous = []
    for source in SOURCES:
        log.info(f"Collecting from: {source['nom']}")
        tous.extend(scrape_generic(source))
        time.sleep(1.0)
    log.info(f"Total raw articles collected: {len(tous)}")
    return tous


# =============================================================================
#  FILTERING
# =============================================================================
#
#  Short, all-caps ASCII acronyms (PMI, FDI, GDP, RCEP, CPTPP...) are prone
#  to false-positive substring matches inside unrelated words. For those,
#  require word boundaries. Longer terms and terms with spaces keep simple
#  substring matching.

def _est_acronyme_ambigu(kw):
    return kw.isascii() and kw.isalpha() and kw.isupper() and len(kw) <= 5


def _compiler_motifs_keywords():
    motifs = []
    for kw in KEYWORDS_ECO_APAC:
        if _est_acronyme_ambigu(kw):
            motifs.append((kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)))
        else:
            motifs.append((kw, None))
    return motifs


KEYWORD_PATTERNS = _compiler_motifs_keywords()


def filtrer_pertinents(articles, vus):
    nouveaux = []
    for a in articles:
        if a["id"] in vus:
            continue
        texte = (a["titre"] + " " + a.get("desc", "")).lower()
        matched = [
            kw for kw, motif in KEYWORD_PATTERNS
            if (motif.search(texte) if motif else kw.lower() in texte)
        ]
        if matched:
            log.info(
                f"  KEPT [{a['source']}] {a['titre'][:70]} "
                f"— matched: {matched[:3]}"
            )
            nouveaux.append(a)
        else:
            log.debug(f"  SKIP [{a['source']}] {a['titre'][:70]}")
    log.info(f"Relevant articles after filtering: {len(nouveaux)}")
    return nouveaux


# =============================================================================
#  ARTICLE ENRICHMENT
# =============================================================================

def enrichir_article(article):
    """Fetch first ~400 chars of article body for better DeepSeek context."""
    try:
        resp = requests.get(
            article["lien"],
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
        article["desc"] = text[:400]
    except Exception as e:
        log.debug(f"Could not enrich {article['lien']}: {e}")
    return article


def enrichir_articles(articles):
    log.info(f"Enriching up to {ENRICH_MAX} articles with body excerpts...")
    enriched = []
    for i, a in enumerate(articles):
        if i < ENRICH_MAX and a["source"] not in (
            "Tavily Search", "DeepSeek Regional Brief"
        ):
            enriched.append(enrichir_article(a))
            time.sleep(0.4)
        else:
            enriched.append(a)
    log.info("Enrichment complete.")
    return enriched


# =============================================================================
#  TAVILY SEARCH — primary channel for local-language / harder-to-reach
#  sources across APAC economies. Query budget is weighted by country tier:
#  primary-tier countries get 2 dedicated queries each, secondary-tier and
#  Oceania countries are grouped into fewer regional queries to keep the
#  total query count manageable.
# =============================================================================

TAVILY_QUERIES = [
    # --- Primary tier — Japan ---
    ("Japan manufacturing PMI economy 2026", "en", "Japan"),
    ("Japan foreign direct investment yen 2026", "en", "Japan"),

    # --- Primary tier — South Korea ---
    ("South Korea manufacturing PMI exports 2026", "en", "South Korea"),
    ("South Korea won interest rate Bank of Korea 2026", "en", "South Korea"),

    # --- Primary tier — India ---
    ("India manufacturing PMI GDP growth 2026", "en", "India"),
    ("India foreign direct investment rupee 2026", "en", "India"),

    # --- Primary tier — Indonesia ---
    ("Indonesia manufacturing PMI economy 2026", "en", "Indonesia"),
    ("Indonesia foreign investment rupiah Bank Indonesia 2026", "en", "Indonesia"),

    # --- Primary tier — Thailand ---
    ("Thailand manufacturing PMI economy 2026", "en", "Thailand"),
    ("Thailand foreign investment baht central bank 2026", "en", "Thailand"),

    # --- Primary tier — Vietnam ---
    ("Vietnam manufacturing PMI exports 2026", "en", "Vietnam"),
    ("Vietnam foreign direct investment China plus one 2026", "en", "Vietnam"),

    # --- Primary tier — Philippines ---
    ("Philippines manufacturing PMI economy 2026", "en", "Philippines"),
    ("Philippines foreign investment peso central bank 2026", "en", "Philippines"),

    # --- Primary tier — Hong Kong ---
    ("Hong Kong economy trade finance 2026", "en", "Hong Kong"),
    ("Hong Kong dollar peg capital flows 2026", "en", "Hong Kong"),

    # --- Primary tier — Taiwan ---
    ("Taiwan manufacturing PMI exports semiconductor 2026", "en", "Taiwan"),
    ("Taiwan dollar foreign investment 2026", "en", "Taiwan"),

    # --- Secondary tier — South Asia (grouped) ---
    ("Bangladesh Pakistan Sri Lanka economy IMF debt 2026", "en", "South Asia"),
    ("Nepal Bhutan economy foreign investment 2026", "en", "South Asia"),

    # --- Secondary tier — Southeast Asia mainland/islands (grouped) ---
    ("Malaysia Singapore manufacturing PMI economy 2026", "en", "Southeast Asia"),
    ("Myanmar Cambodia Laos economy investment 2026", "en", "Southeast Asia"),

    # --- Oceania (grouped) ---
    ("Australia economy interest rate RBA 2026", "en", "Oceania"),
    ("New Zealand economy interest rate RBNZ 2026", "en", "Oceania"),
    ("Pacific islands economy Fiji Papua New Guinea investment 2026", "en", "Oceania"),

    # --- Cross-cutting regional themes ---
    ("Asia supply chain diversification China plus one 2026", "en", "Regional"),
    ("RCEP CPTPP trade agreement Asia Pacific 2026", "en", "Regional"),
    ("Asia currency depreciation central bank intervention 2026", "en", "Regional"),
]


def rechercher_tavily():
    """Search for APAC (ex-Mainland China) economic indicators using Tavily.

    Query allocation is weighted toward primary-tier countries (Japan,
    South Korea, India, Indonesia, Thailand, Vietnam, Philippines, Hong
    Kong, Taiwan), with secondary-tier and Oceania economies covered via
    grouped regional queries.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        log.warning("TAVILY_API_KEY not set — skipping Tavily search.")
        return []

    found     = []
    seen_urls = set()

    for query, lang, scope in TAVILY_QUERIES:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        api_key,
                    "query":          query,
                    "search_depth":   "basic",
                    "max_results":    5,
                    "include_answer": False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data    = resp.json()
            results = data.get("results", [])
            batch   = 0

            for r in results:
                url     = str(r.get("url", "")).strip()
                title   = str(r.get("title", "")).strip()[:150]
                content = str(r.get("content", "")).strip()[:400]

                if not url or not title or url in seen_urls:
                    continue
                seen_urls.add(url)

                found.append({
                    "source": "Tavily Search",
                    "titre":  title,
                    "lien":   url,
                    "desc":   content,
                    "date":   datetime.now().strftime("%Y-%m-%d"),
                    "id":     hashlib.md5((title + url).encode()).hexdigest(),
                })
                batch += 1

            log.info(f"  Tavily [{scope}] '{query[:50]}': {batch} results")

        except Exception as e:
            log.warning(f"Tavily search failed for '{query[:45]}': {e}")

        time.sleep(0.5)

    log.info(f"Tavily total: {len(found)} articles found")
    return found


# =============================================================================
#  DEEPSEEK REGIONAL CONTEXT BRIEF
#
#  Runs on Mondays only. Asks DeepSeek to provide structural context on
#  APAC's (ex-Mainland China) macroeconomic environment from its training
#  knowledge — long-term trends that don't change daily but provide
#  essential background for interpreting weekly signals.
# =============================================================================

ECO_BRIEF_PROMPT = """You are a senior macroeconomic analyst specializing in the 
Asia-Pacific region, EXCLUDING Mainland China (which is covered by a separate brief). 
Your client is TLD Group (Alvest subsidiary), a manufacturer and lessor of Ground 
Support Equipment (GSE) with a regional finance and operations footprint spanning 
Hong Kong, Singapore, Thailand, the Philippines, Japan, South Korea, and Australia, 
plus broader commercial exposure across Southeast Asia, South Asia, and Oceania.

Do NOT cover aviation traffic, airport investment, or airline fleet orders — that is 
handled by a separate GSE-demand-focused brief. Focus strictly on general 
macroeconomic and competitiveness themes.

Based on your training knowledge, provide a structured economic context brief covering:

1. Manufacturing PMI trends across major APAC economies (Japan, South Korea, India, 
   Indonesia, Thailand, Vietnam, Philippines, Taiwan)
2. Foreign direct investment flows and "China+1" supply chain diversification trends
3. Currency trends for major APAC currencies vs USD/EUR and competitiveness implications
4. Interest rate / monetary policy stance across major APAC central banks
5. Key trade agreements and tariff developments affecting the region (RCEP, CPTPP, 
   bilateral deals)
6. Structural reform or regulatory developments material to foreign-invested 
   manufacturers/financial operations in the region
7. Key macroeconomic risks for a company with financial and manufacturing exposure 
   across this region (next 6 months)

For each topic return a JSON object with:
  "topic": topic name
  "status": "FAVORABLE" | "NEUTRAL" | "UNFAVORABLE" for TLD's regional operations
  "trend": "IMPROVING" | "STABLE" | "DETERIORATING"
  "summary": 2-3 sentences of context
  "tld_impact": one sentence on direct impact for TLD's regional finance/operations
  "confidence": "HIGH" | "MEDIUM" | "LOW"

Return ONLY a JSON array of these objects. No markdown fences, no preamble."""


def synthese_regionale_deepseek():
    """Ask DeepSeek for APAC (ex-China) macroeconomic context from its
    training knowledge. Runs Mondays only to avoid redundant daily API calls.
    """
    if datetime.now().weekday() != 0:
        log.info("Regional brief: skipping (runs Mondays only)")
        return []

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — skipping regional brief.")
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info("Requesting regional context brief from DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": ECO_BRIEF_PROMPT}],
            max_tokens=2500,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        text = re.sub(r"```(?:json)?|```", "", text).strip()

        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not array_match:
            log.warning("Regional brief: no JSON array in response")
            return []

        topics = json.loads(array_match.group(0))
        if not isinstance(topics, list):
            return []

        articles = []
        for t in topics:
            if t.get("confidence") == "LOW":
                continue
            topic   = t.get("topic", "")
            status  = t.get("status", "NEUTRAL")
            trend   = t.get("trend", "STABLE")
            summary = t.get("summary", "")
            impact  = t.get("tld_impact", "")

            desc = (
                f"Status for TLD: {status}. Trend: {trend}. "
                f"{summary} TLD impact: {impact}"
            )[:400]

            articles.append({
                "source": "DeepSeek Regional Brief",
                "titre":  f"{topic} — {status} / {trend}"[:150],
                "lien":   "#regional-brief",
                "desc":   desc,
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "id":     hashlib.md5(
                    (topic + datetime.now().strftime("%Y-W%W")).encode()
                ).hexdigest(),
            })

        log.info(f"Regional brief: {len(articles)} topics injected")
        return articles

    except json.JSONDecodeError as e:
        log.warning(f"Regional brief: JSON parse error — {e}")
        return []
    except Exception as e:
        log.warning(f"Regional brief failed: {e}")
        return []


# =============================================================================
#  DEEPSEEK — STRUCTURED ANALYSIS PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a senior macroeconomic analyst advising the CFO APAC of 
TLD Group (Alvest subsidiary), a global manufacturer and lessor of Ground Support 
Equipment (GSE). The CFO's regional finance team spans Hong Kong, Singapore, 
Thailand, the Philippines, Japan, South Korea, and Australia, with commercial 
exposure extending across Southeast Asia, South Asia, and Oceania.

SCOPE: Asia-Pacific EXCLUDING Mainland China (covered by a separate brief).

OUT OF SCOPE — DO NOT GENERATE SIGNALS ON: aviation passenger/cargo traffic, 
airport construction or investment, airline fleet orders, or any GSE-demand 
pull-through signal. These are covered exclusively by a separate GSE-demand 
agent. If an article is purely about aviation/airport demand with no broader 
macroeconomic angle, SKIP it entirely.

YOUR ROLE: translate general macroeconomic, currency, financing, trade, and 
regulatory signals into actionable financial and operational intelligence for 
TLD's regional entities and commercial footprint.

ANALYSIS FRAMEWORK — interpret every signal through these lenses:

MANUFACTURING & SUPPLY CHAIN:
- Regional manufacturing PMI > 52 = expansion; < 48 = contraction — read as a proxy 
  for regional industrial health and supplier/customer capex appetite
- "China+1" supply chain diversification moves (factories relocating to Vietnam, 
  India, Indonesia, etc.) → potential new sourcing or customer-base opportunities, 
  or added complexity for TLD's regional supply chain
- Labour cost trends by country → relative manufacturing competitiveness shifts 
  within the region

FX & FINANCING:
- Local currency depreciation vs EUR/USD → competitiveness of that market's exports, 
  translation impact on TLD's regional entity results reported in EUR
- Local currency appreciation → reverse effect; also affects TLD entity cost base
- Central bank rate hikes → higher financing costs for TLD regional entities and 
  for customers financing GSE leases/purchases in that market
- Rate cuts / monetary easing → cheaper financing, potential demand tailwind
- Credit rating changes / sovereign bond yield moves → country risk premium shifts 
  relevant to treasury and intercompany funding decisions

TRADE & TARIFFS:
- New trade agreements (RCEP, CPTPP, bilateral deals) → potential to ease 
  cross-border operations, customs costs, or component sourcing
- Tariff or export control changes → direct cost impact on any cross-border 
  component or equipment flows
- Trade deficit/surplus shifts → broader signal of a market's external position 
  and currency stability risk

STRUCTURAL REFORM & COMPLIANCE:
- Market liberalization, foreign ownership rule changes, special economic zones → 
  potential for expanded footprint, JV structuring, or new incorporation options
- Regulatory crackdowns, new data protection/cybersecurity laws, capital controls, 
  or repatriation restrictions → direct compliance burden or treasury risk for 
  TLD's regional entities

IMPACT LEVELS:
- CRITICAL: Act within 48h — major threshold breach, urgent opportunity/threat
- IMPORTANT: Act this week — significant shift requiring management attention
- WATCH: Monitor — emerging trend, no immediate action
- INFO: Background context

KEY THRESHOLDS TO FLAG AS CRITICAL:
- PMI < 48 or > 52 for the first time in 3+ months, for any covered country
- Currency move > 3% vs EUR or USD in 30 days
- Policy rate move > 50bps in a single decision
- FDI inflow/outflow swing > 15% YoY for a primary-tier country
- Sovereign credit rating downgrade or default risk escalation
- New capital control or repatriation restriction announced

OUTPUT FORMAT — use EXACTLY this structure:

For each meaningful signal:
===SIGNAL_START===
SIGNAL_ID: [number]
IMPACT: [CRITICAL | IMPORTANT | WATCH | INFO]
COUNTRY: [country or region name, e.g. Vietnam, Hong Kong, Oceania, Regional]
INDICATOR: [PMI | GDP | FDI | FX | MONETARY | TRADE | LABOR | SUPPLY_CHAIN | REFORM | COMPLIANCE | OTHER]
HEADLINE: [One sharp sentence — max 15 words]
READING: [2-3 sentences: what the data shows and what is driving it]
SUPPLY_CHAIN_IMPACT: [2-3 sentences: impact on TLD's regional supply chain, sourcing, or manufacturing footprint outside China — or state "No direct supply chain impact" if not applicable]
FINANCE_FX_IMPACT: [2-3 sentences: impact on TLD's regional entity financing costs, FX exposure, or customer financing conditions in that market]
ACTION: [1-2 sentences: specific recommended action for CFO APAC or regional finance team, time-bound]
===SIGNAL_END===

After ALL signals:
===SUMMARY_START===
EXECUTIVE_SUMMARY: [5-6 sentences for board-level briefing. Overall regional economic picture, key numbers, net impact on TLD APAC operations, priority actions.]
COMPETITIVENESS_OUTLOOK: [2-3 sentences: net regional competitiveness / cost-base outlook for next 30-90 days based on today's signals]
FINANCING_OUTLOOK: [2-3 sentences: regional financing/credit conditions outlook for next 90-180 days]
WATCH_1: [Most critical indicator to monitor this week with specific threshold]
WATCH_2: [Second key indicator with threshold]
WATCH_3: [Third key indicator with threshold]
MAIN_RISK: [Single biggest economic risk for TLD's APAC operations — one sentence]
MAIN_OPPORTUNITY: [Single biggest economic opportunity — one sentence]
===SUMMARY_END===

Rules:
- English only in the output
- Always quantify when possible (%, local currency values, EUR/USD impact, bps)
- No bullet points inside field values — plain prose only
- Skip articles with zero connection to macroeconomics, FX, financing, trade, or 
  regulatory compliance (including any pure aviation/airport/GSE-demand article)
- Always output the SUMMARY block
- Flag threshold breaches explicitly in the HEADLINE using words like ALERT or BREACH
- Always populate COUNTRY — use the most specific country name available, or the 
  grouped region label (e.g. "South Asia", "Oceania", "Regional") if the signal is 
  genuinely multi-country
"""


def construire_prompt_user(articles):
    date_str = datetime.now().strftime("%d %B %Y")
    lines = [
        "APAC ECONOMIC WATCH (EX-MAINLAND CHINA) — TLD Group / Regional Finance",
        f"Date: {date_str}",
        f"Articles to analyze: {len(articles)}",
        "",
    ]
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] SOURCE: {a['source']}")
        lines.append(f"    TITLE: {a['titre']}")
        lines.append(f"    URL: {a['lien']}")
        if a.get("desc"):
            lines.append(f"    EXCERPT: {a['desc'][:350]}")
        lines.append("")

    lines.append(
        "Analyze each article for macroeconomic, FX, financing, trade, or "
        "regulatory signals relevant to TLD Group's APAC (ex-Mainland China) "
        "regional finance and operations. Do NOT generate signals for pure "
        "aviation/airport/GSE-demand content — that is out of scope. "
        "Output ONLY the structured blocks defined in your instructions."
    )
    lines.append("")
    lines.append(
        "CRITICAL RULE: Any article containing specific data points — PMI "
        "readings, FDI figures, currency moves, interest rate decisions, trade "
        "agreement terms, or regulatory changes — MUST generate a signal "
        "regardless of how brief the mention. Quantify every data point you "
        "find. Flag any threshold breach (PMI<48 or >52, FX move>3%, rate "
        "move>50bps, FDI swing>15%) as CRITICAL impact."
    )
    return "\n".join(lines)


# Minimum number of directly-scraped articles guaranteed a slot in the
# DeepSeek batch, so Tavily/regional-brief content never crowds them out
# entirely.
MIN_SCRAPED_QUOTA = 25


def select_balanced_batch(articles, max_total=DEEPSEEK_MAX_ARTICLES, min_scraped=MIN_SCRAPED_QUOTA):
    scraped = [a for a in articles if a["source"] not in ("Tavily Search", "DeepSeek Regional Brief")]
    other = [a for a in articles if a["source"] in ("Tavily Search", "DeepSeek Regional Brief")]

    reserved_scraped = scraped[:min_scraped]
    remaining_slots = max_total - len(reserved_scraped)
    batch = reserved_scraped + other[:remaining_slots]

    if len(batch) < max_total:
        extra = scraped[len(reserved_scraped):len(reserved_scraped) + (max_total - len(batch))]
        batch += extra

    log.info(
        f"Balanced batch: {sum(1 for a in batch if a['source'] not in ('Tavily Search','DeepSeek Regional Brief'))} scraped, "
        f"{sum(1 for a in batch if a['source'] in ('Tavily Search','DeepSeek Regional Brief'))} Tavily/regional-brief "
        f"(of {len(articles)} total relevant articles)"
    )
    return batch[:max_total]


def analyser_avec_deepseek(articles):
    if not articles:
        log.info("No articles to analyze.")
        return "", None

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    batch = select_balanced_batch(articles, DEEPSEEK_MAX_ARTICLES, MIN_SCRAPED_QUOTA)
    if len(articles) > DEEPSEEK_MAX_ARTICLES:
        log.warning(
            f"Capped input at {DEEPSEEK_MAX_ARTICLES} articles "
            f"(had {len(articles)})."
        )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info(f"Sending {len(batch)} articles to DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": construire_prompt_user(batch)},
            ],
            max_tokens=8192,
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        log.info(f"DeepSeek response: {len(raw)} chars, finish_reason={finish_reason}")
        return raw, finish_reason
    except Exception as e:
        log.error(f"DeepSeek API error: {e}")
        return "", None


# =============================================================================
#  PARSER
# =============================================================================

def extract_field(block, field):
    pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)"
    match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parser_analyse(raw_text):
    signals = []
    summary = {
        "executive_summary":      "",
        "competitiveness_outlook":"",
        "financing_outlook":      "",
        "watch":                  [],
        "main_risk":              "",
        "main_opportunity":       "",
    }

    if not raw_text:
        log.warning("Empty DeepSeek response — nothing to parse.")
        return signals, summary

    n_starts    = raw_text.count("===SIGNAL_START===")
    n_ends      = raw_text.count("===SIGNAL_END===")
    has_summary = "===SUMMARY_START===" in raw_text

    if n_starts != n_ends:
        log.warning(
            f"TRUNCATION DETECTED: {n_starts} starts, {n_ends} ends. "
            "Raise max_tokens or reduce input."
        )
    if n_starts > 0 and not has_summary:
        log.warning("TRUNCATION DETECTED: no SUMMARY block found.")

    for block in re.findall(
        r"===SIGNAL_START===(.*?)===SIGNAL_END===", raw_text, re.DOTALL
    ):
        impact = extract_field(block, "IMPACT").upper() or "INFO"
        if impact not in ("CRITICAL", "IMPORTANT", "WATCH", "INFO"):
            impact = "INFO"
        signals.append({
            "id":                   extract_field(block, "SIGNAL_ID"),
            "impact":               impact,
            "country":              extract_field(block, "COUNTRY"),
            "indicator":            extract_field(block, "INDICATOR"),
            "headline":             extract_field(block, "HEADLINE"),
            "reading":              extract_field(block, "READING"),
            "supply_chain_impact":  extract_field(block, "SUPPLY_CHAIN_IMPACT"),
            "finance_fx_impact":    extract_field(block, "FINANCE_FX_IMPACT"),
            "action":               extract_field(block, "ACTION"),
        })

    sm = re.search(
        r"===SUMMARY_START===(.*?)===SUMMARY_END===", raw_text, re.DOTALL
    )
    if sm:
        b = sm.group(1)
        summary["executive_summary"]       = extract_field(b, "EXECUTIVE_SUMMARY")
        summary["competitiveness_outlook"] = extract_field(b, "COMPETITIVENESS_OUTLOOK")
        summary["financing_outlook"]       = extract_field(b, "FINANCING_OUTLOOK")
        summary["main_risk"]               = extract_field(b, "MAIN_RISK")
        summary["main_opportunity"]        = extract_field(b, "MAIN_OPPORTUNITY")
        summary["watch"] = [
            extract_field(b, f"WATCH_{i}")
            for i in range(1, 4)
            if extract_field(b, f"WATCH_{i}")
        ]

    log.info(
        f"Parsed: {len(signals)} signals, "
        f"summary={'yes' if summary['executive_summary'] else 'NO'}"
    )
    return signals, summary


# =============================================================================
#  HTML REPORT
# =============================================================================

IMPACT_CONFIG = {
    "CRITICAL": {"label": "Critical",  "color": "#dc2626", "bg": "#fef2f2",
                 "border": "#fecaca", "text": "#991b1b"},
    "IMPORTANT": {"label": "Important", "color": "#d97706", "bg": "#fffbeb",
                  "border": "#fde68a", "text": "#92400e"},
    "WATCH":     {"label": "Watch",     "color": "#0369a1", "bg": "#f0f9ff",
                  "border": "#bae6fd", "text": "#0c4a6e"},
    "INFO":      {"label": "Info",      "color": "#6b7280", "bg": "#f9fafb",
                  "border": "#e5e7eb", "text": "#374151"},
}

INDICATOR_ICONS = {
    "PMI":          "📊",
    "GDP":          "📈",
    "FDI":          "🌍",
    "FX":           "💱",
    "MONETARY":     "🏦",
    "TRADE":        "🌐",
    "LABOR":        "👷",
    "SUPPLY_CHAIN": "🔗",
    "REFORM":       "🔧",
    "COMPLIANCE":   "⚖️",
    "OTHER":        "📌",
}


def md(text):
    if not text:
        return ""
    html = markdown.markdown(text.strip(), extensions=["nl2br"])
    if html.count("<p>") == 1:
        html = re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.DOTALL)
    return html


def trouver_article(sig, articles):
    haystack = (
        sig.get("headline", "") + " " + sig.get("reading", "")
    ).lower()
    best_article, best_score = None, 0
    for a in articles:
        candidate = (a["titre"] + " " + a.get("desc", "")).lower()
        words = [w for w in re.split(r"[\s\W]+", candidate) if len(w) >= 3]
        score = sum(1 for w in words if w in haystack)
        if score > best_score:
            best_score, best_article = score, a
    return best_article if best_score >= 1 else None


def _render_info_item(sig, articles):
    """Render one collapsed INFO-level item, WITH a clickable source link
    when a matching article can be found."""
    article = trouver_article(sig, articles)
    headline_esc = (
        sig["headline"]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    country_tag = f' <span style="color:#94a3b8;">[{sig["country"]}]</span>' if sig.get("country") else ""
    if article:
        return (
            '<li style="font-size:13px;color:#64748b;padding:3px 0;">'
            f'<a href="{article.get("lien","#")}" target="_blank" rel="noopener" '
            f'style="color:#2563eb;text-decoration:none;">{headline_esc}</a>'
            f'{country_tag}'
            f' <span style="color:#94a3b8;">— {article["source"]}</span>'
            "</li>"
        )
    return f'<li style="font-size:13px;color:#64748b;padding:3px 0;">{headline_esc}{country_tag}</li>'


def render_signal_card(sig, articles):
    cfg  = IMPACT_CONFIG.get(sig["impact"], IMPACT_CONFIG["INFO"])
    icon = INDICATOR_ICONS.get(sig.get("indicator", "OTHER"), "📌")
    article      = trouver_article(sig, articles)
    source_block = ""
    if article and article.get("lien", "#") != "#regional-brief":
        titre_esc = (
            article["titre"]
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        source_block = (
            f'<div class="signal-source">'
            f'<span class="source-label">Source</span>'
            f'<a href="{article.get("lien","#")}" target="_blank" rel="noopener">'
            f"{titre_esc}</a>"
            f'<span class="source-name"> — {article["source"]}</span>'
            f"</div>"
        )

    supply_block = ""
    if sig.get("supply_chain_impact"):
        supply_block = f"""
    <div class="signal-section">
      <div class="signal-section-label">Supply chain impact</div>
      <div class="signal-section-text">{md(sig['supply_chain_impact'])}</div>
    </div>"""

    finance_block = ""
    if sig.get("finance_fx_impact"):
        finance_block = f"""
    <div class="signal-section">
      <div class="signal-section-label">Financing / FX impact</div>
      <div class="signal-section-text">{md(sig['finance_fx_impact'])}</div>
    </div>"""

    country_tag = (
        f'<span class="country-tag">{sig["country"]}</span>'
        if sig.get("country") else ""
    )

    return f"""
<div class="signal-card impact-{sig['impact'].lower()}">
  <div class="signal-card-header" style="border-left:4px solid {cfg['color']};">
    <span class="signal-badge"
          style="background:{cfg['bg']};color:{cfg['text']};border:1px solid {cfg['border']};">
      {cfg['label']}
    </span>
    <span class="indicator-tag">{icon} {sig.get('indicator','')}</span>
    {country_tag}
    <h3 class="signal-headline">{md(sig['headline'])}</h3>
  </div>
  <div class="signal-body">
    <div class="signal-section">
      <div class="signal-section-label">Reading</div>
      <div class="signal-section-text">{md(sig['reading'])}</div>
    </div>
    {supply_block}
    {finance_block}
    <div class="signal-section signal-action">
      <div class="signal-section-label">Recommended action</div>
      <div class="signal-section-text">{md(sig['action'])}</div>
    </div>
    {source_block}
  </div>
</div>"""


def generer_rapport(articles, signals, summary, truncated=False):
    now_full = datetime.now().strftime("%B %d, %Y")
    now_time = datetime.now().strftime("%H:%M")

    counts = {"CRITICAL": 0, "IMPORTANT": 0, "WATCH": 0, "INFO": 0}
    for s in signals:
        counts[s["impact"]] = counts.get(s["impact"], 0) + 1

    actionable  = [s for s in signals if s["impact"] in ("CRITICAL", "IMPORTANT", "WATCH")]
    background  = [s for s in signals if s["impact"] == "INFO"]

    signals_html = ""
    if not actionable and not background:
        signals_html = (
            '<p style="color:#6b7280;font-style:italic;padding:24px 0;">'
            "No significant economic signals identified today.</p>"
        )
    else:
        for sig in actionable:
            signals_html += render_signal_card(sig, articles)
        if background:
            info_items = "".join(
                _render_info_item(sig, articles)
                for sig in background
            )
            signals_html += f"""
<details style="margin-top:12px;">
  <summary style="font-size:12px;color:#94a3b8;cursor:pointer;padding:8px 4px;
                  user-select:none;list-style:none;">
    <span style="font-size:10px;background:#f1f5f9;border:1px solid #e2e8f0;
                 border-radius:20px;padding:2px 8px;color:#64748b;font-weight:600;">
      + {len(background)} background item{"s" if len(background)!=1 else ""}
    </span>
    <span style="color:#94a3b8;margin-left:8px;">— no immediate action, click to expand</span>
  </summary>
  <ul style="list-style:none;padding:12px 16px;margin-top:8px;
             background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;">
    {info_items}
  </ul>
</details>"""

    counter_html = "".join(
        f'<span class="counter-pill" '
        f'style="background:{IMPACT_CONFIG[lvl]["bg"]};'
        f'color:{IMPACT_CONFIG[lvl]["text"]};'
        f'border:1px solid {IMPACT_CONFIG[lvl]["border"]};">'
        f'{counts[lvl]} {IMPACT_CONFIG[lvl]["label"]}</span>'
        for lvl in ("CRITICAL", "IMPORTANT", "WATCH", "INFO")
        if counts[lvl] > 0
    )

    watch_html   = "".join(f"<li>{md(w)}</li>" for w in summary.get("watch", []))
    exec_html    = md(summary.get("executive_summary", ""))
    comp_html    = md(summary.get("competitiveness_outlook", ""))
    fin_html     = md(summary.get("financing_outlook", ""))
    risk_html    = md(summary.get("main_risk", ""))
    opp_html     = md(summary.get("main_opportunity", ""))
    sources_list = "".join(f"<li>{s['nom']}</li>" for s in SOURCES)

    trunc_banner = ""
    if truncated:
        trunc_banner = """
<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;
            padding:12px 16px;margin-bottom:24px;font-size:13px;color:#713f12;">
  <strong>Warning:</strong> DeepSeek confirmed its response was cut off
  (finish_reason=length) — some signals or the summary block are likely
  missing. Reduce DEEPSEEK_MAX_ARTICLES; deepseek-chat's output ceiling is a
  hard limit, so raising max_tokens further will not help.
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APAC Economic Watch (ex-China) — {now_full}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --ink:#0f172a;--ink-2:#334155;--ink-3:#64748b;--ink-4:#94a3b8;
  --surface:#fff;--surface-1:#f8fafc;--border:#e2e8f0;
  --radius:8px;--radius-lg:12px;
}}
body{{font-family:'Inter',-apple-system,sans-serif;background:#f0f2f5;
      color:var(--ink);line-height:1.6;padding:32px 16px 64px}}
.wrapper{{max-width:960px;margin:0 auto}}

/* MASTHEAD */
.masthead{{background:var(--ink);border-radius:var(--radius-lg) var(--radius-lg) 0 0;padding:28px 36px 24px}}
.masthead-eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.12em;
                   text-transform:uppercase;color:#64748b;margin-bottom:8px}}
.masthead-title{{font-size:22px;font-weight:600;letter-spacing:-.02em;color:#fff;margin-bottom:4px}}
.masthead-subtitle{{font-size:13px;color:#475569;margin-bottom:12px}}
.masthead-meta{{display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.meta-item{{font-size:13px;color:#94a3b8;display:flex;align-items:center;gap:6px}}
.meta-item strong{{color:#e2e8f0;font-weight:500}}
.masthead-counters{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;
                    padding-top:16px;border-top:1px solid #1e293b}}
.counter-pill{{font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;letter-spacing:.02em}}

/* CARD BODY */
.card-body{{background:var(--surface);border:1px solid var(--border);border-top:none;
            border-radius:0 0 var(--radius-lg) var(--radius-lg);padding:36px}}
.section-header{{display:flex;align-items:center;gap:10px;margin-bottom:20px;
                 padding-bottom:12px;border-bottom:1px solid var(--border)}}
.section-header h2{{font-size:13px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:var(--ink-3)}}
.section-divider{{margin:36px 0;border:none;border-top:1px solid var(--border)}}

/* EXEC SUMMARY */
.exec-panel{{background:var(--ink);border-radius:var(--radius-lg);padding:24px 28px;
             margin-bottom:24px;color:#e2e8f0;font-size:15px;line-height:1.75}}
.exec-panel-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;
                   text-transform:uppercase;color:#475569;margin-bottom:10px}}
.exec-panel p{{margin:0}}

/* OUTLOOK PANELS */
.outlook-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}}
.outlook-card{{border-radius:var(--radius);padding:16px 18px;border:1px solid}}
.outlook-card.margins{{background:#f0fdf4;border-color:#bbf7d0}}
.outlook-card.demand{{background:#eff6ff;border-color:#bfdbfe}}
.outlook-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
                margin-bottom:8px}}
.outlook-card.margins .outlook-label{{color:#166534}}
.outlook-card.demand .outlook-label{{color:#1e40af}}
.outlook-text{{font-size:13px;line-height:1.6}}
.outlook-card.margins .outlook-text{{color:#14532d}}
.outlook-card.demand .outlook-text{{color:#1e3a8a}}
.outlook-text p{{margin:0}}

/* SIGNAL CARDS */
.signal-card{{border:1px solid var(--border);border-radius:var(--radius-lg);
              margin-bottom:16px;overflow:hidden;transition:box-shadow .15s}}
.signal-card:hover{{box-shadow:0 4px 12px rgba(0,0,0,.06)}}
.signal-card-header{{padding:14px 20px;background:var(--surface-1);
                     display:flex;align-items:flex-start;gap:10px;flex-wrap:wrap}}
.signal-badge{{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;
               white-space:nowrap;letter-spacing:.03em;margin-top:2px;flex-shrink:0}}
.indicator-tag{{font-size:11px;font-weight:500;color:var(--ink-3);
                background:#f1f5f9;padding:3px 8px;border-radius:6px;
                white-space:nowrap;margin-top:2px;flex-shrink:0}}
.country-tag{{font-size:11px;font-weight:600;color:#1e40af;
             background:#eff6ff;padding:3px 8px;border-radius:6px;
             white-space:nowrap;margin-top:2px;flex-shrink:0}}
.signal-headline{{font-size:15px;font-weight:600;color:var(--ink);line-height:1.4;flex:1}}
.signal-headline p{{margin:0}}
.signal-body{{padding:20px;display:grid;gap:14px}}
.signal-section-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                        letter-spacing:.1em;color:var(--ink-4);margin-bottom:4px}}
.signal-section-text{{font-size:14px;color:var(--ink-2);line-height:1.65}}
.signal-section-text p{{margin:0}}
.signal-action .signal-section-text{{color:var(--ink);font-weight:500}}
.signal-source{{padding-top:10px;border-top:1px dashed var(--border);font-size:12px;
                color:var(--ink-4);display:flex;flex-wrap:wrap;gap:4px;align-items:center}}
.source-label{{font-weight:600;text-transform:uppercase;letter-spacing:.06em;
               font-size:10px;color:var(--ink-4);margin-right:4px}}
.signal-source a{{color:#2563eb;text-decoration:none;font-weight:500}}
.signal-source a:hover{{text-decoration:underline}}
.source-name{{color:var(--ink-4)}}

/* WATCH / RISK / OPP */
.watch-panel{{background:#fffbeb;border:1px solid #fde68a;border-radius:var(--radius-lg);
              padding:18px 22px;margin-bottom:12px}}
.watch-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:#92400e;margin-bottom:10px}}
.watch-panel ol{{padding-left:20px;display:grid;gap:6px}}
.watch-panel li{{font-size:14px;color:#78350f;line-height:1.5}}
.watch-panel li p{{margin:0}}
.risk-panel{{background:#fef2f2;border:1px solid #fecaca;border-radius:var(--radius-lg);
             padding:16px 22px;margin-bottom:12px}}
.risk-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                   letter-spacing:.08em;color:#991b1b;margin-bottom:6px}}
.risk-panel-text{{font-size:14px;color:#7f1d1d;font-weight:500;line-height:1.6}}
.risk-panel-text p{{margin:0}}
.opp-panel{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:var(--radius-lg);
            padding:16px 22px}}
.opp-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.08em;color:#166534;margin-bottom:6px}}
.opp-panel-text{{font-size:14px;color:#14532d;font-weight:500;line-height:1.6}}
.opp-panel-text p{{margin:0}}

/* SOURCES */
.sources-panel{{background:var(--surface-1);border:1px solid var(--border);
                border-radius:var(--radius);padding:14px 18px;margin-top:32px}}
.sources-panel-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                       letter-spacing:.1em;color:var(--ink-4);margin-bottom:8px}}
.sources-panel ul{{list-style:none;display:flex;flex-wrap:wrap;gap:4px 0;
                   column-gap:24px;columns:2}}
.sources-panel li{{font-size:12px;color:var(--ink-3);break-inside:avoid}}
.sources-panel li::before{{content:"·";margin-right:6px;color:var(--ink-4)}}

/* FOOTER */
.page-footer{{text-align:center;font-size:11px;color:var(--ink-4);margin-top:24px;
              font-family:'IBM Plex Mono',monospace;letter-spacing:.04em}}

@media(max-width:640px){{
  body{{padding:12px 8px 48px}}
  .masthead,.card-body{{padding:20px 16px}}
  .outlook-grid{{grid-template-columns:1fr}}
  .sources-panel ul{{columns:1}}
}}
</style>
</head>
<body>
<div class="wrapper">

<div class="masthead">
  <div class="masthead-eyebrow">TLD Group · Alvest · CFO APAC Intelligence</div>
  <div class="masthead-title">APAC Economic Watch (ex-Mainland China)</div>
  <div class="masthead-subtitle">Manufacturing PMI · FDI &amp; supply chain · FX &amp; financing · Trade &amp; reform · Compliance</div>
  <div class="masthead-meta">
    <div class="meta-item"><span>Date</span><strong>{now_full}</strong></div>
    <div class="meta-item"><span>Generated</span><strong>{now_time}</strong></div>
    <div class="meta-item"><span>Articles analyzed</span><strong>{len(articles)}</strong></div>
    <div class="meta-item"><span>Signals</span><strong>{len(signals)}</strong></div>
  </div>
  {f'<div class="masthead-counters">{counter_html}</div>' if counter_html else ''}
</div>

<div class="card-body">

  {trunc_banner}

  {f'<div class="exec-panel"><div class="exec-panel-label">Executive summary</div>{exec_html}</div>' if exec_html else ''}

  {f'''<div class="outlook-grid">
    <div class="outlook-card margins">
      <div class="outlook-label">🌏 Regional competitiveness outlook (30-90 days)</div>
      <div class="outlook-text">{comp_html}</div>
    </div>
    <div class="outlook-card demand">
      <div class="outlook-label">🏦 Financing / credit outlook (90-180 days)</div>
      <div class="outlook-text">{fin_html}</div>
    </div>
  </div>''' if comp_html or fin_html else ''}

  <div class="section-header"><h2>Signals</h2></div>
  {signals_html}

  {'<hr class="section-divider">' if watch_html or risk_html or opp_html else ''}

  {f'<div class="section-header"><h2>To watch this week</h2></div><div class="watch-panel"><div class="watch-panel-label">Key indicators &amp; thresholds</div><ol>{watch_html}</ol></div>' if watch_html else ''}

  {f'<div class="risk-panel"><div class="risk-panel-label">Main risk</div><div class="risk-panel-text">{risk_html}</div></div>' if risk_html else ''}

  {f'<div class="opp-panel"><div class="opp-panel-label">Main opportunity</div><div class="opp-panel-text">{opp_html}</div></div>' if opp_html else ''}

  <div class="sources-panel">
    <div class="sources-panel-label">Monitored sources (scraped + Tavily)</div>
    <ul>
      {sources_list}
      <li>Tavily Search (country-tiered queries, primary + secondary + Oceania)</li>
    </ul>
  </div>

</div>

<div class="page-footer">
  APAC Economic Watch (ex-Mainland China) · TLD Group / Alvest · Powered by DeepSeek + Tavily · {now_full}
</div>

</div>
</body>
</html>"""


# =============================================================================
#  SAVE
# =============================================================================

def sauvegarder_rapport(rapport_html):
    # "reports" : dossier scanné par weekly_digest_agent.py sur tous les
    # repos. Nom de fichier avec date ISO pour le filtrage "7 derniers
    # jours" du digest, + une copie reports/latest.html en repli.
    dossier = Path("reports")
    dossier.mkdir(exist_ok=True, parents=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    fichier = dossier / f"apac_eco_watch_report_{date_str}.html"
    with open(fichier, "w", encoding="utf-8") as f:
        f.write(rapport_html)
    (dossier / "latest.html").write_text(rapport_html, encoding="utf-8")
    log.info(f"Report saved: {fichier.absolute()} (and reports/latest.html)")
    return fichier


# =============================================================================
#  MAIN
# =============================================================================

def executer_agent():
    log.info("=" * 60)
    log.info("Starting APAC Economic Watch Agent (ex-Mainland China) v1.0")
    log.info("=" * 60)
    try:
        vus = charger_vus()

        # 1. Scraped sources (international/regional English-language outlets)
        tous_articles = collecter_tous_articles()

        # 2. Tavily search — primary channel for local-language / harder to
        #    reach sources, weighted by country tier.
        tavily_articles = rechercher_tavily()
        if tavily_articles:
            tous_articles.extend(tavily_articles)
            log.info(f"Total after Tavily: {len(tous_articles)} articles")

        # 3. DeepSeek regional context brief (Mondays only)
        regional_articles = synthese_regionale_deepseek()
        if regional_articles:
            tous_articles.extend(regional_articles)
            log.info(f"Total after regional brief: {len(tous_articles)} articles")

        # 4. Filter by keywords
        articles_pertinents = filtrer_pertinents(tous_articles, vus)

        # 5. Prioritize: Tavily first, regional brief second, scraped last
        #    Ensures harder-to-reach country data is never dropped by the
        #    DeepSeek cap.
        def source_priority(a):
            if a["source"] == "Tavily Search":
                return 0
            if a["source"] == "DeepSeek Regional Brief":
                return 1
            return 2

        articles_pertinents.sort(key=source_priority)
        log.info(
            f"After prioritization: "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'Tavily Search')} Tavily, "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'DeepSeek Regional Brief')} regional-brief, "
            f"{sum(1 for a in articles_pertinents if a['source'] not in ('Tavily Search','DeepSeek Regional Brief'))} scraped"
        )

        # 6. Enrich scraped articles (Tavily/regional-brief already have content)
        if articles_pertinents:
            articles_pertinents = enrichir_articles(articles_pertinents)

        # 7. Analyze with DeepSeek
        raw_analyse, finish_reason = (
            analyser_avec_deepseek(articles_pertinents)
            if articles_pertinents
            else ("", None)
        )

        # 8. Save raw output for debugging
        Path("reports").mkdir(exist_ok=True, parents=True)
        Path("reports/debug_raw_apac_eco.txt").write_text(
            raw_analyse or "", encoding="utf-8"
        )
        log.info("Raw DeepSeek output saved to reports/debug_raw_apac_eco.txt")

        # 9. Truncation detection
        # api_truncated (finish_reason=="length") is authoritative and the
        # only signal that drives the report's alarming banner. format_mismatch
        # can happen even in short responses from an isolated formatting slip
        # on one item — logged, but not treated with the same severity.
        n_starts  = raw_analyse.count("===SIGNAL_START===")
        n_ends    = raw_analyse.count("===SIGNAL_END===")
        has_sum   = "===SUMMARY_START===" in raw_analyse
        api_truncated   = (finish_reason == "length")
        format_mismatch = (n_starts != n_ends) or (n_starts > 0 and not has_sum)

        if api_truncated:
            log.warning(
                "TRUNCATION CONFIRMED BY API: finish_reason=length. "
                "Reduce DEEPSEEK_MAX_ARTICLES (currently %d).", DEEPSEEK_MAX_ARTICLES,
            )
        elif format_mismatch:
            log.warning(
                f"Formatting mismatch (NOT a real truncation — response was "
                f"only {len(raw_analyse)} chars): {n_starts} SIGNAL_START vs "
                f"{n_ends} SIGNAL_END, summary_present={has_sum}. See "
                f"reports/debug_raw_apac_eco.txt to find the exact spot."
            )

        truncated = api_truncated  # only the authoritative signal drives the report banner

        # 10. Parse
        signals, summary = parser_analyse(raw_analyse)

        # 11. Generate report
        rapport_html = generer_rapport(
            articles_pertinents, signals, summary, truncated=truncated
        )

        # 12. Save
        fichier = sauvegarder_rapport(rapport_html)
        print(f"✅ Report generated: {fichier}")

        # 13. Mark as seen
        for a in articles_pertinents:
            vus.add(a["id"])
        sauvegarder_vus(vus)
        log.info("Done.")

    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    executer_agent()

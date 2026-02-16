"""Extrator de features para detecção de phishing via ML."""

import math
from collections import Counter
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tldextract import extract
from re import search

SUSPICIOUS_TLDS = [
    'zip', 'mov', 'xyz', 'top', 'info', 'live', 'gq',
    'cf', 'tk', 'ml', 'work', 'link', 'click', 'surf',
    'gdn', 'buzz', 'rest',
]

PHISH_KEYWORDS = [
    'login', 'verify', 'account', 'security', 'update',
    'confirm', 'banco', 'caixa', 'signin', 'ebayisapi', 'webscr',
]

SHORTENERS = [
    'bit.ly', 't.co', 'tinyurl.com', 'is.gd', 'soo.gd',
    'ow.ly', 'buff.ly', 'rebrand.ly', 'shorturl.at',
]

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
}

CONTENT_FEATURE_DEFAULTS = {
    'nb_hyperlinks': -1, 'ratio_intHyperlinks': -1,
    'ratio_extHyperlinks': -1, 'ratio_nullHyperlinks': -1,
    'nb_extCSS': -1, 'login_form': -1,
    'external_favicon': -1, 'links_in_tags': -1,
    'submit_email': -1, 'sfh': -1, 'iframe': -1,
    'popup_window': -1, 'safe_anchor': -1,
    'onmouseover': -1, 'right_clic': -1,
    'empty_title': -1, 'domain_in_title': -1,
}


def extract_url_structure_features(url: str, hostname: str, path: str, extracted) -> dict:
    """Extrai features da estrutura/contagem de caracteres da URL.
    Requer: url completa, hostname, path e resultado do tldextract.
    Retorna: dict com ~18 features numéricas.
    """
    return {
        'length_url': len(url),
        'length_hostname': len(hostname),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': url.count('@'),
        'nb_qm': url.count('?'),
        'nb_and': url.count('&'),
        'nb_eq': url.count('='),
        'nb_underscore': url.count('_'),
        'nb_slash': url.count('/'),
        'nb_www': 1 if 'www' in extracted.subdomain else 0,
        'nb_com': url.count('com'),
        'nb_dslash': url.count('//'),
        'http_in_path': 1 if 'http' in path.lower() else 0,
        'https_token': 1 if 'https' in hostname else 0,
        'ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if url else 0,
        'ratio_digits_host': sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0,
        'punycode': 1 if 'xn--' in hostname else 0,
    }


def extract_port_and_tld_features(parsed, extracted, hostname: str) -> dict:
    """Extrai features de porta não-padrão e TLD em locais incomuns.
    Requer: resultado do urlparse, tldextract e hostname.
    Retorna: dict com 4 features binárias.
    """
    path = parsed.path or ''
    return {
        'port': 1 if parsed.port and parsed.port not in (80, 443) else 0,
        'tld_in_path': 1 if extracted.suffix and extracted.suffix in path else 0,
        'tld_in_subdomain': 1 if extracted.suffix and extracted.suffix in extracted.subdomain else 0,
        'abnormal_subdomain': 1 if extracted.subdomain and extracted.subdomain != 'www' and '.' in extracted.subdomain else 0,
    }


def extract_domain_word_features(url: str, hostname: str, path: str, extracted) -> dict:
    """Extrai features de palavras, domínio e subdomínios.
    Requer: url, hostname, path e resultado do tldextract.
    Retorna: dict com ~14 features.
    """
    subdomains = extracted.subdomain.split('.') if extracted.subdomain else []
    words_host = hostname.split('.') if hostname else ['']
    words_path = [w for w in path.replace('/', ' ').split() if w]
    hints = sum(kw in url.lower() for kw in PHISH_KEYWORDS)

    return {
        'nb_subdomains': len(subdomains),
        'prefix_suffix': 1 if '-' in extracted.domain else 0,
        'shortening_service': 1 if hostname in SHORTENERS else 0,
        'nb_redirection': url.count('//') - 1,
        'nb_external_redirection': 0,
        'char_repeat': max(Counter(url.lower()).values()) if url else 0,
        'shortest_word_host': min(len(w) for w in words_host) if words_host else 0,
        'longest_word_host': max(len(w) for w in words_host) if words_host else 0,
        'avg_word_host': sum(len(w) for w in words_host) / len(words_host) if words_host else 0,
        'avg_word_path': sum(len(w) for w in words_path) / len(words_path) if words_path else 0,
        'phish_hints': hints,
        'suspecious_tld': 1 if extracted.suffix in SUSPICIOUS_TLDS else 0,
        'random_domain': _is_random_domain(extracted.domain),
        'ip': 1 if _has_ip(url) else 0,
    }


def _is_random_domain(domain: str) -> int:
    """Retorna 1 se o domínio tem alta entropia (aparência aleatória)."""
    if not domain:
        return 0
    freq = Counter(domain.lower())
    probs = [c / len(domain) for c in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return 1 if entropy > 3.5 else 0


def _has_ip(url: str) -> bool:
    """Retorna True se a URL contém um endereço IP."""
    return search(r"(\d{1,3}\.){3}\d{1,3}", url) is not None


def extract_content_features(url: str, domain: str) -> tuple[dict, bool]:
    """Extrai features do HTML da página via request.
    Requer: url completa e domínio base.
    Retorna: (dict de features, True se acessível / False se não).
    """
    try:
        response = requests.get(url, timeout=7, headers=REQUEST_HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = response.text.lower()

        features = {}
        features.update(_extract_hyperlink_features(soup, domain))
        features.update(_extract_form_features(soup, page_text, domain))
        features.update(_extract_security_features(soup, page_text, domain))
        features.update(_extract_title_features(soup, domain))
        return features, True
    except Exception:
        return dict(CONTENT_FEATURE_DEFAULTS), False


def _extract_hyperlink_features(soup: BeautifulSoup, domain: str) -> dict:
    """Extrai ratios de links internos, externos e nulos."""
    all_links = soup.find_all('a', href=True)
    nb = len(all_links)
    int_links = ext_links = null_links = 0

    for a in all_links:
        href = a['href']
        if href.startswith('#') or href.startswith('javascript') or href == '':
            null_links += 1
        elif domain in href or href.startswith('/'):
            int_links += 1
        else:
            ext_links += 1

    return {
        'nb_hyperlinks': nb,
        'ratio_intHyperlinks': int_links / nb if nb > 0 else 0,
        'ratio_extHyperlinks': ext_links / nb if nb > 0 else 0,
        'ratio_nullHyperlinks': null_links / nb if nb > 0 else 0,
        'safe_anchor': 1 if nb > 0 and null_links / nb > 0.3 else 0,
    }


def _extract_form_features(soup: BeautifulSoup, page_text: str, domain: str) -> dict:
    """Extrai features de formulários: login, SFH, mailto."""
    forms = soup.find_all('form')

    has_login = any(
        'password' in str(f).lower() or 'login' in str(f).lower() or 'signin' in str(f).lower()
        for f in forms
    )
    sfh_suspicious = any(
        f.get('action', '') in ('', 'about:blank')
        or (f.get('action', '').startswith('http') and domain not in f.get('action', ''))
        for f in forms
    )

    return {
        'login_form': 1 if has_login else 0,
        'sfh': 1 if sfh_suspicious else 0,
        'submit_email': 1 if 'mailto:' in page_text or 'mail(' in page_text else 0,
    }


def _extract_security_features(soup: BeautifulSoup, page_text: str, domain: str) -> dict:
    """Extrai features de segurança: iframe, popup, favicon externo, CSS externo."""
    css_links = soup.find_all('link', rel='stylesheet')
    ext_css = sum(1 for c in css_links if c.get('href') and domain not in c.get('href', ''))

    favicons = soup.find_all('link', rel=lambda r: r and 'icon' in r.lower() if r else False)
    ext_fav = any(domain not in (f.get('href') or '') for f in favicons)

    return {
        'nb_extCSS': ext_css,
        'external_favicon': 1 if ext_fav else 0,
        'links_in_tags': len(soup.find_all('link')) + len(soup.find_all('script', src=True)),
        'iframe': 1 if soup.find_all('iframe') else 0,
        'popup_window': 1 if 'window.open' in page_text else 0,
        'onmouseover': 1 if 'onmouseover' in page_text else 0,
        'right_clic': 1 if 'event.button==2' in page_text or 'contextmenu' in page_text else 0,
    }


def _extract_title_features(soup: BeautifulSoup, domain: str) -> dict:
    """Extrai features do título: vazio e domínio presente."""
    title = soup.title.string if soup.title else ''
    return {
        'empty_title': 1 if not title or title.strip() == '' else 0,
        'domain_in_title': 1 if title and domain.lower() in title.lower() else 0,
    }


class PhishingAnalyzer:
    """Extrator de features para o modelo ML de phishing."""

    def analyze(self, url: str) -> dict:
        """Extrai todas as features ML de uma URL.
        Requer: URL completa (com http/https).
        Retorna: dict com 'ml_features'.
        """
        parsed = urlparse(url)
        extracted = extract(url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        domain = f"{extracted.domain}.{extracted.suffix}"

        ml_features = {}
        ml_features.update(extract_url_structure_features(url, hostname, path, extracted))
        ml_features.update(extract_port_and_tld_features(parsed, extracted, hostname))
        ml_features.update(extract_domain_word_features(url, hostname, path, extracted))

        content_features, _ = extract_content_features(url, domain)
        ml_features.update(content_features)

        return {"ml_features": ml_features}

"""알레르기 그룹별 다국어 낱말(en/fr/es/de/ko). 정규식 조각, 소문자 기준."""
import re
from typing import List
from .model import AllergyProfile, Evidence, ReadLine

_W = r"(?<![a-zà-ÿœ])"  # 단어 시작
GROUPS = {
    "milk": [r"milk", _W + r"lait(?!ue)", r"leche", r"milch", r"lact", r"whey", r"suero", r"molke",
             r"cas[eé]in", r"kasein", _W + r"butter", r"beurre", r"mantequilla", _W + r"cream",
             r"cr[eè]me", _W + r"nata\b", r"yogh?urt", r"yaourt", r"cheese", r"fromage", r"queso",
             r"käse", r"우유", r"유청", r"유크림", r"분유", r"락토", r"카제인", r"버터", r"치즈", r"유당", r"크림"],
    "gluten": [_W + r"wheat", r"gluten", _W + r"barley", _W + r"rye\b", _W + r"oats?\b", r"spelt",
               _W + r"malt(?!o|ol)", r"\bbl[eé]\b", r"froment", r"seigle", r"orge\b", r"avoine", r"[eé]peautre",
               r"trigo", r"cebada", r"centeno", r"avena", r"espelta", r"weizen", r"gerste", r"roggen",
               r"hafer", r"dinkel", r"밀(?!크)", r"소맥", r"보리", r"호밀", r"귀리", r"글루텐", r"맥아"],
    "egg": [_W + r"eggs?\b", _W + r"egg[ -]", r"oeufs?\b", r"œufs?\b", r"huevos?", r"\bei\b", r"\beier\b",
            r"eigelb", r"eiwei", r"vollei", r"albumin", r"계란", r"달걀", r"난각", r"난백", r"난황", r"\b난\b"],
    "soy": [_W + r"soy", _W + r"soja", r"edamame", r"tofu", r"대두", r"(?<!땅)콩", r"간장", r"된장", r"두부"],
    "nut": [r"peanut", r"groundnut", r"cacahu", r"erdnu", _W + r"nuts?\b", r"almond", r"amande", r"almendra",
            r"mandel", r"hazelnut", r"noisette", r"avellana", r"haselnu", r"walnut", _W + r"noix", r"nuez",
            r"walnu", r"cashew", r"pistach", r"pecan", r"macadamia", r"fruits? à coque", r"frutos? secos",
            r"nüsse", r"땅콩", r"아몬드", r"호두", r"견과", r"캐슈", r"피스타치오", r"헤이즐넛", r"\b잣"],
    "fish": [_W + r"fish", r"poisson", r"pescado", r"fisch", r"anchov", _W + r"tuna", r"\bthon\b", r"atún",
             r"salmon", r"saumon", r"\bcod\b", r"surimi", r"생선", r"어육", r"멸치", r"참치", r"연어", r"명태", r"액젓"],
    "other": [r"crustac", r"shrimp", r"prawn", _W + r"crab", r"lobster", r"crevette", r"gamba", r"garnele",
              r"새우", r"크랩", r"sulph?ite", r"sulfite", r"sulph?ur dioxide", r"dioxyde de soufre",
              r"sulfuroso", r"schwefeldioxid", r"mustard", r"moutarde", r"mostaza", r"senf", r"겨자", r"sesame",
              r"s[eé]same", r"s[eé]samo", r"sesam", r"참깨", r"gelatin", r"g[eé]latine", r"젤라틴", r"celery",
              r"c[eé]leri", r"apio", r"sellerie", r"lupin", r"mollus", r"mussel", r"squid", r"오징어"],
}
ALL_GROUPS = tuple(GROUPS)
_COMPILED = {g: re.compile("|".join(f"(?:{p})" for p in ps), re.IGNORECASE) for g, ps in GROUPS.items()}

# Open Food Facts 태그 → 그룹 (평가 전용 매핑)
OFF_TAG_GROUP = {
    "en:milk": "milk", "en:gluten": "gluten", "en:eggs": "egg", "en:soybeans": "soy",
    "en:peanuts": "nut", "en:nuts": "nut", "en:fish": "fish", "en:crustaceans": "other",
    "en:sulphur-dioxide-and-sulphites": "other", "en:mustard": "other", "en:sesame-seeds": "other",
    "en:gelatin": "other", "es:avena": "gluten", "es:Avena": "gluten",
}


def match_allergens(lines: List[ReadLine], profile: AllergyProfile) -> List[Evidence]:
    out: List[Evidence] = []
    for g in profile.groups:
        rx = _COMPILED.get(g)
        if rx is None:
            continue
        for i, ln in enumerate(lines):
            m = rx.search(ln.text)
            if m:
                out.append(Evidence(group=g, term=m.group(0), line_index=i, line_text=ln.text))
                break  # 그룹당 첫 근거 줄
    return sorted(out, key=lambda e: (e.line_index, e.group))

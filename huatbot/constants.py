"""Fixed facts about Singapore Pools TOTO and 4D.

Prize percentages and amounts here are the built in fallback. At run time
``prize_rules.load_prize_rules`` tries to confirm them against the official
prize structure pages and reports whether it could.
"""
from __future__ import annotations

from datetime import time
from math import comb

# Site endpoints

BASE = "https://www.singaporepools.com.sg"
ARCHIVE = f"{BASE}/DataFileArchive/Lottery/Output"

TOTO_DRAW_LIST_URL = f"{ARCHIVE}/toto_result_draw_list_en.html"
FOURD_DRAW_LIST_URL = f"{ARCHIVE}/fourd_result_draw_list_en.html"

TOTO_RESULT_URL = f"{BASE}/en/product/sr/Pages/toto_results.aspx?sppl={{sppl}}"
FOURD_RESULT_URL = f"{BASE}/en/product/Pages/4d_results.aspx?sppl={{sppl}}"

TOTO_NEXT_DRAW_URL = f"{ARCHIVE}/toto_next_draw_estimate_en.html"
FOURD_NEXT_DRAW_URL = f"{ARCHIVE}/fourd_next_draw_info_en.html"

TOTO_CASCADE_LIST_URL = f"{ARCHIVE}/toto_result_cascade_draw_list_en.html"
TOTO_HONGBAO_LIST_URL = f"{ARCHIVE}/toto_result_hongbao_draw_list_en.html"
TOTO_SPECIAL_LIST_URL = f"{ARCHIVE}/toto_result_special_draw_list_en.html"

TOTO_PRIZE_RULES_URL = "https://online2.singaporepools.com/en/lottery/toto-prize-structure"
FOURD_PRIZE_RULES_URL = "https://online2.singaporepools.com/en/lottery/4d-prize-structure"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)

# TOTO (6 from 49 format, since draw 2995 on 9 Oct 2014)

TOTO_FIRST_CURRENT_FORMAT_DRAW = 2995
TOTO_MAX_NUMBER = 49
TOTO_PICK = 6
TOTO_COMBOS = comb(49, 6)  # 13,983,816
TOTO_LOW_MAX = 24  # low half is 1 to 24, high half is 25 to 49
TOTO_BIRTHDAY_MAX = 31

# Number of the 13,983,816 boards that land in each prize group.
TOTO_GROUP_COMBOS = {
    1: 1,                                   # 6 numbers
    2: comb(6, 5) * 1,                      # 5 + additional = 6
    3: comb(6, 5) * 42,                     # 5 = 252
    4: comb(6, 4) * 1 * 42,                 # 4 + additional = 630
    5: comb(6, 4) * comb(42, 2),            # 4 = 12,915
    6: comb(6, 3) * 1 * comb(42, 2),        # 3 + additional = 17,220
    7: comb(6, 3) * comb(42, 3),            # 3 = 229,600
}
TOTO_ANY_PRIZE_COMBOS = sum(TOTO_GROUP_COMBOS.values())  # 260,624, about 1 in 54

TOTO_POOL_SHARE_OF_SALES = 0.54
TOTO_GROUP_POOL_PCT = {1: 0.38, 2: 0.08, 3: 0.055, 4: 0.03}
TOTO_FIXED_PRIZES = {5: 50.0, 6: 25.0, 7: 10.0}
TOTO_MIN_GROUP1 = 1_000_000.0
TOTO_BOARD_COST = 1.0
TOTO_SNOWBALL_LIMIT = 4  # Group 1 cascades after the 4th draw in a row with no winner

# System n bet = every 6 number board inside n numbers.
TOTO_SYSTEM_BOARDS = {n: comb(n, 6) for n in range(7, 13)}  # 7:7, 8:28, 9:84, 10:210, 11:462, 12:924

TOTO_DRAW_TYPES = ("normal", "cascade", "hongbao", "special")

# 4D

FOURD_TIERS = ("first", "second", "third", "starter", "consolation")
FOURD_TIER_COUNTS = {"first": 1, "second": 1, "third": 1, "starter": 10, "consolation": 10}
FOURD_NUMBERS_PER_DRAW = 23
FOURD_SPACE = 10_000

# Prize per $1 stake.
FOURD_PRIZES = {
    "big": {"first": 2000.0, "second": 1000.0, "third": 490.0, "starter": 250.0, "consolation": 60.0},
    "small": {"first": 3000.0, "second": 2000.0, "third": 800.0},
}
FOURD_POSITIONS = ("thousands", "hundreds", "tens", "units")

# Schedule (Singapore time)

SG_TZ_NAME = "Asia/Singapore"
TOTO_WEEKDAYS = (0, 3)  # Mon, Thu
FOURD_WEEKDAYS = (2, 5, 6)  # Wed, Sat, Sun
DRAW_TIME = time(18, 30)
DEFAULT_RUN_TIME = time(19, 30)
DEFAULT_RETRY_MINUTES = 10
DEFAULT_RETRY_HOURS = 2.0

TELEGRAM_MAX_CHARS = 4000

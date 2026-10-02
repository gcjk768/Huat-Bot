"""Render toto.csv rows into Singapore Pools style HTML, plus a fake fetcher.

The pages use the selectors verified on the live site (th.drawDate, th.drawNumber,
td.win1..6, td.additional, td.jackpotPrize, table.tableWinningShares) wrapped in
realistic noise: a head with scripts, a nav, unrelated tables, decoy classes and
newlines inside cells. Parsers that pass on these pages are selector based, not
position based.

Public API (also used by the integration tests):
    toto_result_html(row)
    draw_list_html(rows_or_pairs), draw_type_list_html(draw_numbers)
    toto_next_draw_html(dt, jackpot, hint=None)
    toto_prize_structure_html(rendered=True)
    FakeFetcher(pages), fake_site(toto_df, next_toto=..., cascade=..., ...)
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from huatbot import constants as C
from huatbot.fetch import toto_result_url
from huatbot.http import FetchError
from huatbot.parse import sppl

# formatting like the site


def date_text(value: Any) -> str:
    """"Thu, 01 Oct 2026" from a date, datetime, Timestamp or ISO string."""
    d = pd.Timestamp(value)
    return d.strftime("%a, %d %b %Y")


def money_text(value: Any) -> str:
    """"$1,185,926" (cents only when needed); NaN or None -> "-"."""
    if value is None or pd.isna(value):
        return "-"
    x = float(value)
    return f"${x:,.0f}" if x.is_integer() else f"${x:,.2f}"


def count_text(value: Any) -> str:
    """Number of winning shares; 0 is shown as "-" like the site does."""
    n = int(value or 0)
    return "-" if n == 0 else f"{n:,}"


# page wrapper with noise

_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>{title}</title>
  <link rel="stylesheet" href="/_layouts/15/SingaporePools/css/bootstrap.min.css">
  <script type="text/javascript">
    // decoys: markup inside scripts must never be parsed as cells
    var tpl = "<td class='win1'>99</td><th class='drawNumber'>Draw No. 1</th>";
    var jackpotBanner = "$9,999,999";
    window.dataLayer = window.dataLayer || [];
  </script>
  <style>.win1 {{ color: red; }} td.jackpotPrize {{ font-weight: bold; }}</style>
</head>
<body class="ms-backgroundImage">
  <form method="post" action="./results.aspx" id="aspnetForm">
  <div class="aspNetHidden"><input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="/wEPDwUKMTY1NDU2MTA1MmRk" /></div>
  <header class="navbar navbar-default">
    <ul class="nav navbar-nav">
      <li><a href="/en/product/Pages/4d_results.aspx">4D Results</a></li>
      <li><a href="/en/product/sr/Pages/toto_results.aspx">TOTO Results</a></li>
      <li><a href="/en/product/Pages/sweep_results.aspx">Singapore Sweep</a></li>
    </ul>
    <div class="win1-banner">Win up to $12,000,000 this Hong Bao season!</div>
  </header>
  <table class="table promo">
    <tr><th>Promotion</th><th>Group 1</th></tr>
    <tr><td class="promoCell">Group 1</td><td>$5,000,000</td><td>99</td></tr>
  </table>
  <div class="container main">
"""

_FOOT = """
  </div>
  <footer>
    <p>Copyright &copy; Singapore Pools (Private) Limited. All rights reserved.</p>
    <table class="footer-links"><tr><td>Responsible Gaming</td><td>Contact Us</td></tr></table>
  </footer>
  </form>
  <script type="text/javascript">
    $(document).ready(function () {{ $('.drawDate').css('font-weight', 'bold'); }});
  </script>
</body>
</html>
"""


def _page(title: str, body: str) -> str:
    return _HEAD.format(title=title) + body + _FOOT.format()


# result pages


def toto_result_html(row: Mapping | pd.Series) -> str:
    """A TOTO result page for one toto.csv row."""
    draw = int(row["draw_number"])
    nums = [int(row[f"n{i}"]) for i in range(1, 7)]
    win_cells = "\n".join(
        f"          <td class='win{i}'>\n            {n}\n          </td>" for i, n in enumerate(nums, start=1)
    )
    share_rows = "\n".join(
        f"        <tr>\n          <td>Group {g}</td>\n          <td>{money_text(row[f'g{g}_share'])}</td>\n"
        f"          <td> {count_text(row[f'g{g}_winners'])} </td>\n        </tr>"
        for g in range(1, 8)
    )
    body = f"""
    <div class="tables-wrap">
      <table class='table table-striped'>
        <thead>
          <tr>
            <th class='drawDate'>
              {date_text(row['draw_date'])}
            </th>
            <th class='drawNumber'>Draw No. {draw}</th>
          </tr>
        </thead>
      </table>
      <table class='table table-striped'>
        <thead><tr><th colspan='6'>Winning Numbers</th></tr></thead>
        <tbody>
          <tr>
{win_cells}
          </tr>
        </tbody>
      </table>
      <table class='table table-striped'>
        <thead><tr><th>Additional Number</th></tr></thead>
        <tbody><tr><td class='additional'> {int(row['additional'])} </td></tr></tbody>
      </table>
      <table class='table table-striped jackpotPrizeTable'>
        <thead><tr><th>Group 1 Prize</th></tr></thead>
        <tbody><tr><td class='jackpotPrize'>{money_text(row['jackpot'])}</td></tr></tbody>
      </table>
      <table class='table table-striped tableWinningShares'>
        <thead><tr><th colspan='3'>Winning Shares</th></tr></thead>
        <tbody>
        <tr>
          <th>Prize Group</th>
          <th>Share Amount</th>
          <th>No. of Winning Shares</th>
        </tr>
{share_rows}
        </tbody>
      </table>
    </div>
    <table class="table recent"><tr><td>Group 2</td><td>$88,888</td><td>7</td></tr></table>
"""
    return _page("TOTO Results", body)


# draw lists


def _pairs(rows_or_pairs: Any) -> list[tuple[int, Any]]:
    """Normalise a DataFrame, rows, (number, date) pairs or bare numbers to (number, date) pairs."""
    if isinstance(rows_or_pairs, pd.DataFrame):
        return [(int(n), d) for n, d in zip(rows_or_pairs["draw_number"], rows_or_pairs["draw_date"])]
    out = []
    for item in rows_or_pairs:
        if isinstance(item, (pd.Series, Mapping)):
            out.append((int(item["draw_number"]), item.get("draw_date")))
        elif isinstance(item, (tuple, list)):
            out.append((int(item[0]), item[1] if len(item) > 1 else None))
        else:
            out.append((int(item), None))
    return out


def draw_list_html(rows_or_pairs: Any, value_as_sppl: bool = False, select_class: str = "selectDrawList") -> str:
    """A draw list page: one <option> per draw, newest first.

    ``value_as_sppl`` writes the sppl string into the value attribute instead of the number,
    to exercise the decoding path of the parser.
    """
    options = []
    for number, when in sorted(_pairs(rows_or_pairs), key=lambda p: p[0], reverse=True):
        text = date_text(when) if when is not None and not pd.isna(when) else f"Draw {number}"
        value = sppl(number) if value_as_sppl else str(number)
        options.append(f"  <option querystring='sppl={sppl(number)}' value='{value}'>{text}</option>")
    return (
        f"<select name='drawList' class='form-control {select_class}' "
        "onchange='LoadDrawResult(this)'>\n" + "\n".join(options) + "\n</select>\n"
    )


def draw_type_list_html(draw_numbers: Iterable[Any]) -> str:
    """A cascade, Hongbao or special draw list: same layout as the draw list.

    Items may be draw numbers or (number, date) pairs.
    """
    return draw_list_html(list(draw_numbers), select_class="selectDrawTypeList")


# next draw pages


def _time_text(dt: datetime, style: str = "dot") -> str:
    hour = dt.hour % 12 or 12
    if style == "colon":
        return f"{hour}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"
    if style == "upper":
        return f"{hour}.{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"
    return f"{hour}.{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"


def _when_text(dt: Any, time_style: str) -> str:
    if dt is None:
        return "To be announced"
    if isinstance(dt, datetime):
        return f"{date_text(dt)} , {_time_text(dt, time_style)}"
    return f"{date_text(dt)} , 6.30pm"


_HINT_TEXT = {"cascade": "Cascade Draw", "hongbao": "Hong Bao Draw", "special": "Special Draw"}


def toto_next_draw_html(dt: Any, jackpot: float | None, hint: str | None = None, time_style: str = "dot") -> str:
    """The TOTO next draw estimate fragment. ``time_style``: "dot" 6.30pm, "colon" 6:30pm, "upper" 6.30 PM."""
    amount = money_text(jackpot) + " est" if jackpot is not None else "To be announced"
    hint_html = f"\n  <div class='drawTypeBanner'><b>{_HINT_TEXT[hint]}</b></div>" if hint else ""
    return f"""<div class='divNextDrawJackpot'>
  <script type='text/javascript'>var previousJackpot = '$7,777,777';</script>
  <div class='row'>
    <div class='col-xs-12'><p class='text-center'>Next Jackpot</p></div>
  </div>
  <div class='row'>
    <div class='col-xs-12'>
      <span style='color:#EC243D; font-weight:bold;'>
        {amount}
      </span>
    </div>
  </div>{hint_html}
  <div class='row'>
    <div class='col-xs-12'><p class='text-center'>Next Draw</p></div>
    <div class='col-xs-12'><p class='toto-draw-date'>{_when_text(dt, time_style)}</p></div>
  </div>
</div>
"""


# prize structure page (online2)

_JS_SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title} | Singapore Pools</title>
<script src="/static/js/main.4f2a9c.js"></script>
<script>window.__INITIAL_STATE__ = {{"page": "{slug}"}};</script>
</head>
<body><noscript>You need to enable JavaScript to run this app.</noscript><div id="root"></div></body></html>
"""


def toto_prize_structure_html(rendered: bool = True, group_pool_pct: Mapping[int, float] | None = None,
                              fixed_prizes: Mapping[int, float] | None = None,
                              pool_share: float | None = C.TOTO_POOL_SHARE_OF_SALES,
                              min_group1: float = C.TOTO_MIN_GROUP1) -> str:
    """The TOTO prize structure page. rendered=False gives the JavaScript shell with no figures."""
    if not rendered:
        return _JS_SHELL.format(title="TOTO Prize Structure", slug="toto-prize-structure")
    pct = dict(C.TOTO_GROUP_POOL_PCT if group_pool_pct is None else group_pool_pct)
    fixed = dict(C.TOTO_FIXED_PRIZES if fixed_prizes is None else fixed_prizes)
    matched = {1: "6 Winning Numbers", 2: "5 Winning Numbers + Additional Number", 3: "5 Winning Numbers",
               4: "4 Winning Numbers + Additional Number", 5: "4 Winning Numbers",
               6: "3 Winning Numbers + Additional Number", 7: "3 Winning Numbers"}

    def amount(g: int) -> str:
        if g in pct:
            text = f"{pct[g] * 100:g}% of Prize Pool"
            if g == 1:
                text += f" (minimum {money_text(min_group1)})"
            return text
        return f"{money_text(fixed[g])} per winning combination"

    rows = "\n".join(
        f"      <tr><td>Group {g}</td><td>{matched[g]}</td><td>{amount(g)}</td></tr>" for g in range(1, 8)
    )
    share_text = (f"<p>The Prize Pool for each draw is {pool_share * 100:g}% of the total sales for that draw.</p>"
                  if pool_share is not None else "")
    body = f"""
    <h1>TOTO Prize Structure</h1>
    <p>Choose 6 numbers from 1 to 49. Each $1 bet gives you one combination.</p>
    {share_text}
    <table class="prize-structure">
      <thead><tr><th>Prize Group</th><th>Winning Numbers Matched</th><th>Prize Amount</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
    <p>Group 1 prize snowballs up to 4 draws if there is no winner.</p>
"""
    return _page("TOTO Prize Structure", body)


# fake fetcher


class FakeFetcher:
    """Offline stand in for http.Fetcher: serves a dict url -> html and records every request.

    A value may also be an exception instance, which is raised for that URL. Unknown
    URLs raise FetchError (like a 404).
    """

    def __init__(self, pages: Mapping[str, Any] | None = None) -> None:
        self.pages: dict[str, Any] = dict(pages or {})
        self.requested: list[str] = []
        self._lock = threading.Lock()

    def get(self, url: str) -> str:
        with self._lock:
            self.requested.append(url)
        if url not in self.pages:
            raise FetchError("page not found (HTTP 404)", url=url, status=404, attempts=1)
        value = self.pages[url]
        if isinstance(value, BaseException):
            raise value
        return value

    def get_many(self, urls: list[str]) -> dict[str, str | FetchError]:
        out: dict[str, str | FetchError] = {}
        for url in dict.fromkeys(urls):
            try:
                out[url] = self.get(url)
            except FetchError as exc:
                out[url] = exc
        return out

    def count(self, url: str) -> int:
        """How many times ``url`` was requested."""
        with self._lock:
            return self.requested.count(url)

    def close(self) -> None:
        pass


def fake_site(
    toto_df: pd.DataFrame | None = None,
    *,
    next_toto: Any = None,
    cascade: Iterable[int] | None = None,
    hongbao: Iterable[int] | None = None,
    special: Iterable[int] | None = None,
    list_size: int = 150,
    prize_pages: str | None = "rendered",
) -> FakeFetcher:
    """A FakeFetcher serving the whole TOTO site for the given history.

    Every row gets a result page; the draw list holds the newest ``list_size`` draws.
    Draw type lists default to the draw_type column of ``toto_df``. The next draw page
    comes from ``next_toto`` (models.NextToto) or from huatbot.synth when omitted.
    ``prize_pages``: "rendered", "js" (JavaScript shell) or None (page missing).
    """
    from huatbot.synth import synth_next_toto

    pages: dict[str, Any] = {}
    if toto_df is not None and len(toto_df):
        for _, row in toto_df.iterrows():
            pages[toto_result_url(int(row["draw_number"]))] = toto_result_html(row)
        pages[C.TOTO_DRAW_LIST_URL] = draw_list_html(toto_df.sort_values("draw_number").tail(list_size))
        dates = dict(zip(toto_df["draw_number"].astype(int), toto_df["draw_date"]))
        for name, given, url in (("cascade", cascade, C.TOTO_CASCADE_LIST_URL),
                                 ("hongbao", hongbao, C.TOTO_HONGBAO_LIST_URL),
                                 ("special", special, C.TOTO_SPECIAL_LIST_URL)):
            if given is None:
                given = toto_df.loc[toto_df["draw_type"] == name, "draw_number"]
            pages[url] = draw_type_list_html([(int(n), dates.get(int(n))) for n in given])
        nt = next_toto if next_toto is not None else synth_next_toto(toto_df)
        pages[C.TOTO_NEXT_DRAW_URL] = toto_next_draw_html(nt.draw_datetime, nt.jackpot_estimate,
                                                          getattr(nt, "draw_type_hint", None))
    if prize_pages:
        pages[C.TOTO_PRIZE_RULES_URL] = toto_prize_structure_html(rendered=prize_pages == "rendered")
    return FakeFetcher(pages)

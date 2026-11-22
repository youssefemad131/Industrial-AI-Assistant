"""FactoryLens AI - conversation engine (orchestrator + answer builder + validator).

Design (Week 5 flow): every path ends in an answer, a recovery or a person.
 * All numbers come from pandas calculations on the synthetic CSVs (never from the model).
 * Every answer has three blocks: Observed fact / Hypothesis / Missing data,
   plus source, time range and a next check (Explainability principle).
 * Validator blocks any number that was not computed (E4 -> regenerate, then facts-only).
 * Optional LLM polish (USE_LLM=1) may only reword; validator rejects invented numbers.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LOG_FILE = Path(os.getenv("LOG_FILE", Path(__file__).parent.parent / "logs" / "chat_log.jsonl"))
WEEKS = {"previous": ("2026-10-18", "2026-10-22"), "latest": ("2026-10-25", "2026-10-29")}
SYNTH_NOTE = "All data is synthetic (illustrative). Not real factory data."

# ----------------------------------------------------------------- messages
FALLBACK = {
    "en": ("I cannot answer that with the data I have. I can help with defect rates, shift comparisons, "
           "data coverage and possible causes from our maintenance notes. Try one of these questions, "
           "or talk to a person."),
    "ar": ("لا أستطيع الإجابة عن ذلك بالبيانات المتاحة لدي. يمكنني المساعدة في معدلات العيوب، ومقارنة الورديات، "
           "وتغطية البيانات، والأسباب المحتملة من ملاحظات الصيانة. جرّب أحد هذه الأسئلة أو تحدّث إلى شخص."),
}
HANDOVER = {
    "en": ("I am passing this to {who}. I have prepared a summary so you do not need to repeat yourself: "
           "your question, what the data shows, and what is missing."),
    "ar": ("سأحوّل هذا إلى {who}. جهّزت ملخصًا حتى لا تضطر إلى التكرار: سؤالك، وما تُظهره البيانات، وما هو ناقص."),
}
LABELS = {
    "en": {"fact": "Observed fact", "hyp": "Hypothesis", "miss": "Missing data",
           "src": "Source", "time": "Time range", "next": "Next check"},
    "ar": {"fact": "Observed fact · حقيقة مرصودة", "hyp": "Hypothesis · فرضية", "miss": "Missing data · بيانات ناقصة",
           "src": "Source · المصدر", "time": "Time range · الفترة", "next": "Next check · الخطوة التالية"},
}
SUGGESTED = ["Did defects go up this week?", "Compare the shifts", "What data do you use?"]
PERIOD_REPLIES = ["Latest week (25-29 Oct)", "Previous week (18-22 Oct)"]
ASK = {"en": "Which period should I look at?", "ar": "ما الفترة التي تريد أن أنظر فيها؟"}
ERROR_MSG = {"en": "Something went wrong on my side. I kept your question. You can retry or talk to a person.",
             "ar": "حدث خطأ من جانبي. احتفظت بسؤالك. يمكنك إعادة المحاولة أو التحدث إلى شخص."}

# ----------------------------------------------------------------- intent rules (order matters)
AR = r"[\u0600-\u06FF]"
RULES = [
    ("real_data", r"real (elaraby|factory|company|production)\b.{0,25}(data|numbers|rates?|defects?)|actual (elaraby|factory) (data|numbers)"),
    ("unsafe_misuse", r"ignore (all |your |previous )*(instructions|rules)|system prompt|make up (some )?(numbers|data|figures)|"
                      r"invent (some )?(numbers|data|figures|a source)|fake\b.{0,25}(report|numbers|data|results)|falsif|"
                      r"hide (the )?(defects?|records?|results?)|delete (the )?(logs?|records?)|"
                      r"bypass|disable (the )?(\w+ )?(interlock|safety|alarm)|"
                      r"تجاهل (كل )?التعليمات|اخترع|زوّر|زور"),
    ("handover_person", r"(talk|speak|connect|pass) (me )?to (a |the )?(person|human|supervisor|someone|agent)|"
                        r"real person|human agent|أريد (التحدث|التكلم) (إلى|مع) (شخص|مشرف)|عايز (اتكلم|أكلم) (مع )?(حد|مشرف)"),
    ("handover_confirm", r"(confirm|prove|certify|sign off|approve)\b.*\b(cause|root cause|oven|supplier|production|release|batch|shipment|decision)|"
                         r"\bdefinitely\b|release (the |this )?batch|"
                         r"(أكد|اعتمد).*(السبب|الدفعة)"),
    ("handover_worker", r"(which|what|who)\b.*\b(worker|operator|employee|technician)\b|"
                        r"\bblame\b|who is (responsible|to blame|at fault)|(worst|slowest|lazy) (worker|operator|employee)|"
                        r"أي (عامل|موظف)|مين (العامل|الموظف)"),
    ("handover_live", r"\b(sap|live (data|machine)|real[- ]time|right now|current machine status|machine status now)\b"),
    ("greeting", r"^\s*(hi|hello|hey|good (morning|evening)|start)\b|^\s*help\s*[?!.]*\s*$|^\s*(مرحبا|أهلا|اهلا|السلام عليكم|هاي)"),
    ("data_limits", r"what data|data (do|are) you|which data|where (do|does) .*(numbers?|data)|your sources?|how reliable|"
                    r"limits?\b|coverage|what can you do|بيانات|مصادر"),
    ("types", r"defect types?|types? of defects?|which defect|what defect|most common defect|نوع العيب|أنواع العيوب"),
    ("investigate", r"\bwhy\b|investigate|reason|root cause|what (caused|is causing)|explain|لماذا|ليه|سبب|تحقيق"),
    ("shifts", r"shift|وردية|ورديات"),
    ("quick", r"defect|reject|scrap|went up|go up|increase|rise|rate|how many|عيوب|عيب|معدل|ارتفع"),
]
PERIOD_LATEST = r"this week|latest week|current week|هذا الأسبوع|الأسبوع (الحالي|الأخير)|25"
PERIOD_PREV = r"last week|previous week|prior week|الأسبوع (الماضي|السابق)|18"
PERIOD_OUT = (r"last month|this month|month|september|august|january|quarter|\byear\b|2025|yesterday|today|"
              r"الشهر|سبتمبر|العام")


def lang_of(text: str) -> str:
    return "ar" if re.search(AR, text) else "en"


# ----------------------------------------------------------------- ledger + validator
class Ledger:
    """Registers every number that was actually computed, so the validator can check the text."""
    def __init__(self):
        self.vals: set[str] = set()

    def c(self, n) -> str:           # count
        n = int(n)
        self.vals.add(str(n))
        return f"{n:,}"

    def p(self, x) -> str:           # percent
        self.vals.add(f"{float(x):.1f}")
        return f"{float(x):.1f}%"

    def quote(self, text: str) -> str:   # text copied from a data field: its numbers are "from the data"
        for t in re.findall(r"\d+(?:\.\d+)?", str(text)):
            self.vals.add(t)
        return text

    def f(self, x) -> str:           # plain decimal (e.g. average minutes)
        self.vals.add(f"{float(x):.1f}")
        return f"{float(x):.1f}"


_STRIP = [r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{1,2}(?:[–-]\d{1,2})? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*(?: \d{4})?\b",
          r"\b\d{1,2}[–-]\d{1,2}\b", r"\b[A-Za-z]{1,4}-\d+(?:-\d+)*\b", r"\b[Pp]\d\b", r"\b[Ss]hift \d\b", r"\b[Ss]hifts? \d(?:, \d)*\b",
          r"\b\d+ (?:C|°C)\b", r"\bnoted\b"]


def validate_numbers(text: str, ledger: Ledger, extra_ok: set[str] | None = None) -> list[str]:
    """Return numbers in `text` that were never computed. Empty list = valid."""
    t = text
    for pat in _STRIP:
        t = re.sub(pat, " ", t)
    found = [x.replace(",", "") for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t)]
    ok = ledger.vals | (extra_ok or set())
    return [x for x in found if x not in ok]


# ----------------------------------------------------------------- engine
class FactoryLens:
    def __init__(self, data_dir: Path = DATA_DIR):
        rd = lambda n: pd.read_csv(data_dir / f"{n}.csv")
        self.insp, self.prod = rd("inspection_records"), rd("shift_production")
        self.maint, self.lots = rd("maintenance_events"), rd("supplier_lots")
        self.sessions: dict[str, dict] = {}

    # ---- helpers
    @staticmethod
    def _wk(df, wk):
        a, b = WEEKS[wk]
        return df[(df["date"] >= a) & (df["date"] <= b)]

    @staticmethod
    def _rate(df):
        d, n = int(df.defect_qty.sum()), int(df.inspected_qty.sum())
        return d, n, (round(100 * d / n, 1) if n else None)

    @staticmethod
    def _rng(wk):
        a, b = map(pd.Timestamp, WEEKS[wk])
        return f"{a.day}–{b.day} {b.strftime('%b %Y')}"

    def _group_rates(self, df, col):
        out = {}
        for k, g in df.groupby(col):
            out[k] = self._rate(g)
        return out

    # ---- answer envelope
    def _env(self, intent, lang, facts=(), hyps=(), miss=(), source="", time_range="", nxt="", quick=None,
             ledger: Ledger | None = None, note=SYNTH_NOTE):
        return dict(type="answer", intent=intent, lang=lang, facts=list(facts), hypotheses=list(hyps),
                    missing=list(miss), source=source, time_range=time_range, next_check=nxt,
                    quick_replies=quick or [], note=note, _ledger=ledger or Ledger())

    def render(self, a: dict) -> str:
        if a["type"] != "answer" or not (a["facts"] or a["hypotheses"] or a["missing"]):
            return a["reply"]
        L = LABELS[a["lang"]]
        out = [L["fact"]] + [f"- {x}" for x in a["facts"]]
        out += [L["hyp"]] + [f"- {x}" for x in (a["hypotheses"] or ["None offered for this question."])]
        out += [L["miss"]] + [f"- {x}" for x in a["missing"]]
        out += [f"{L['src']}: {a['source']}", f"{L['time']}: {a['time_range']}", f"{L['next']}: {a['next_check']}",
                a["note"]]
        return "\n".join(out)

    # ---- missing-data (always visible)
    def _gaps(self, df, L=None):
        out = ["Humidity and paint viscosity are not recorded in any table.",
               "Oven temperature is not logged; only operator notes exist."]
        n = int(((df.defect_qty > 0) & df.defect_type.isna()).sum())
        if n:
            n = L.c(n) if L else n
            out.append(f"{n} inspection record(s) in this period have defects but no defect type.")
        return out

    # ---- answer builders -----------------------------------------------------------------
    def a_quick(self, lang):
        L = Ledger()
        B, A = self._wk(self.insp, "latest"), self._wk(self.insp, "previous")
        dB, nB, rB = self._rate(B); dA, nA, rA = self._rate(A)
        move = "up" if rB > rA else "down" if rB < rA else "unchanged"
        facts = [f"Defect rate is {move}: {L.p(rB)} in the latest week ({self._rng('latest')}) vs {L.p(rA)} in the previous week ({self._rng('previous')}).",
                 f"Counts: {L.c(dB)} defective of {L.c(nB)} inspected (latest); {L.c(dA)} of {L.c(nA)} (previous)."]
        n = int(((B.defect_qty > 0) & B.defect_type.isna()).sum())
        miss = self._gaps(B, L)
        return self._env("quick_check", lang, facts, [], miss,
                         f"inspection_records ({L.c(len(B) + len(A))} records)",
                         f"{self._rng('previous')} and {self._rng('latest')}",
                         "Ask “Why did it rise?” to start a guided investigation, or “Compare the shifts”.",
                         ["Why did it rise?", "Compare the shifts"], L)

    def a_shifts(self, lang):
        L = Ledger()
        B, A = self._wk(self.insp, "latest"), self._wk(self.insp, "previous")
        gb, ga = self._group_rates(B, "shift"), self._group_rates(A, "shift")
        facts = []
        for s in sorted(gb):
            facts.append(f"Shift {s}: {L.p(gb[s][2])} latest week ({L.c(gb[s][0])} of {L.c(gb[s][1])}) vs {L.p(ga[s][2])} previous week ({L.c(ga[s][0])} of {L.c(ga[s][1])}).")
        worst = max(gb, key=lambda s: gb[s][2])
        facts.append(f"Shift {worst} has the highest rate in the latest week.")
        return self._env("shift_comparison", lang, facts, [], self._gaps(B, L),
                         f"inspection_records ({L.c(len(B) + len(A))} records)",
                         f"{self._rng('previous')} and {self._rng('latest')}",
                         "A higher rate on one shift is not proof of a shift effect. Ask “Why did it rise?” to see possible causes.",
                         ["Why did it rise?", "Which defect types?"], L)

    def a_types(self, lang):
        L = Ledger()
        B = self._wk(self.insp, "latest")
        d = B[B.defect_qty > 0].dropna(subset=["defect_type"]).groupby("defect_type").defect_qty.sum().sort_values(ascending=False)
        tot = int(d.sum())
        facts = [f"{k}: {L.c(v)} defects ({L.p(100 * v / tot)} of typed defects)." for k, v in d.items()]
        return self._env("defect_types", lang, facts, [], self._gaps(B, L),
                         f"inspection_records ({L.c(len(B))} records)", self._rng("latest"),
                         "Ask “Compare the shifts” to see where these defects concentrate.",
                         ["Compare the shifts"], L)

    def a_data(self, lang):
        L = Ledger()
        facts = [f"inspection_records: {L.c(len(self.insp))} rows (batch, shift, machine, defect type, inspected and defect quantity, supplier lot).",
                 f"shift_production: {L.c(len(self.prod))} shift summaries (units, downtime).",
                 f"maintenance_events: {L.c(len(self.maint))} events with notes (the only source of hypotheses).",
                 f"supplier_lots: {L.c(len(self.lots))} lots with inspection status."]
        miss = ["Humidity, paint viscosity and oven temperature logs are not recorded.",
                "No operator or worker data is stored or used.",
                "No live SAP or machine connection; data covers 18–29 Oct 2026 only."]
        return self._env("data_and_limits", lang, facts, [], miss,
                         "backend/data/*.csv (synthetic, team-created)", f"{self._rng('previous')} and {self._rng('latest')}",
                         "Ask a question about defect rates, shifts or possible causes.", SUGGESTED, L)

    def a_investigate(self, lang, period="both"):
        L = Ledger()
        B, A = self._wk(self.insp, "latest"), self._wk(self.insp, "previous")
        dA, nA, rA = self._rate(A)
        if period == "previous":
            facts = [f"Defect rate was {L.p(rA)} in the previous week ({self._rng('previous')}): {L.c(dA)} of {L.c(nA)} inspected."]
            miss = ["There is no week before 18 Oct in the data, so I cannot say whether the rate rose or fell.",
                    "I did not look for causes: there is no baseline to explain."] + self._gaps(A, L)[:2]
            return self._env("investigation_no_baseline", lang, facts, [], miss,
                             f"inspection_records ({L.c(len(A))} records)", self._rng("previous"),
                             "Pick the latest week to compare it with the previous week.", ["Latest week (25-29 Oct)"], L)
        dB, nB, rB = self._rate(B)
        gs, gm = self._group_rates(B, "shift"), self._group_rates(B, "machine_id")
        ws, wm = max(gs, key=lambda s: gs[s][2]), max(gm, key=lambda m: gm[m][2])
        om = [m for m in gm if m != wm][0]
        facts = [f"Defect rate: {L.p(rB)} in the latest week ({self._rng('latest')}) vs {L.p(rA)} in the previous week ({self._rng('previous')}).",
                 f"Highest shift: shift {ws} at {L.p(gs[ws][2])} ({L.c(gs[ws][0])} of {L.c(gs[ws][1])}); other shifts are lower.",
                 f"Highest line: {wm} at {L.p(gm[wm][2])} vs {om} at {L.p(gm[om][2])}."]
        typed = B[B.defect_qty > 0].dropna(subset=["defect_type"]).groupby("defect_type").defect_qty.sum().sort_values(ascending=False)
        if len(typed):
            facts.append(f"Most common defect type: {typed.index[0]} ({L.c(typed.iloc[0])} of {L.c(typed.sum())} typed defects).")
        pb, pa = self._wk(self.prod, "latest"), self._wk(self.prod, "previous")
        dB3, dA3 = pb[pb["shift"] == ws].downtime_min.mean(), pa[pa["shift"] == ws].downtime_min.mean()
        facts.append(f"Average downtime on shift {ws}: {L.f(dB3)} min per shift (latest) vs {L.f(dA3)} min (previous).")
        # hypotheses: only from maintenance notes / supplier status that exist in the data
        hyps, hyp_src = [], []
        ev = self.maint
        ev = ev[(ev.machine_id == wm) & (ev.status.isin(["open", "noted"]))]
        for _, e in ev.iterrows():
            strong = (pd.notna(e["shift"]) and int(e["shift"]) == ws)
            hyps.append(f"{e.event_id} ({e.date}, line {e.machine_id}): “{L.quote(e.notes)}” Oven temperature drift may contribute to the rise. "
                        f"Evidence: {'moderate' if strong else 'weak'} (timing and line match{' and same shift' if strong else ''}); a timing match is not proof of cause.")
            hyp_src.append(e.event_id)
        lot_id = B[B.machine_id == wm].supplier_lot.mode().iloc[0]
        lot = self.lots[self.lots.lot_id == lot_id].iloc[0]
        miss = self._gaps(B, L)
        if lot.inspection_status != "passed":
            lw = B[B.supplier_lot == lot_id]; lo = B[B.supplier_lot != lot_id]
            _, nl, rl = self._rate(lw); _, no, ro = self._rate(lo)
            facts.append(f"Latest week, records using lot {lot_id}: {L.p(rl)} ({L.c(nl)} inspected) vs {L.p(ro)} for other lots ({L.c(no)} inspected).")
            hyps.append(f"Paint lot {lot_id} ({lot.supplier_code}, delivered {lot.delivery_date}) may contribute. Evidence: weak (same period only; the lot has no incoming inspection).")
            miss.insert(0, f"Lot {lot_id} has no incoming inspection record (status: {lot.inspection_status.replace('_', ' ')}).")
            hyp_src.append(lot_id)
        return self._env("guided_investigation", lang, facts, hyps, miss,
                         f"inspection_records, shift_production, maintenance_events ({', '.join(hyp_src)}), supplier_lots",
                         f"{self._rng('previous')} and {self._rng('latest')}",
                         f"You choose: check oven temperature on line {wm} (shift {ws}) with a calibrated thermometer, or ask QC to inspect lot {lot_id}. I do not confirm causes.",
                         ["Compare the shifts", "Talk to a person"], L)

    def a_partial(self, lang):
        a = self.a_quick(lang)
        a["intent"] = "partial_data"
        a["missing"].insert(0, "The period you asked for is not in the data. I only have 18–29 Oct 2026 (two weeks), so I show those.")
        a["next_check"] = "Choose one of the two weeks, or ask for a person to pull older records."
        return a

    # ---- non-answer replies ---------------------------------------------------------------
    def _msg(self, intent, lang, text, quick=None, **kw):
        return dict(type=kw.pop("type", "message"), intent=intent, lang=lang, reply=text, facts=[], hypotheses=[],
                    missing=[], quick_replies=quick or [], _ledger=Ledger(), **kw)

    def _greeting(self, lang):
        if lang == "ar":
            t = ("أهلًا، أنا FactoryLens AI. أساعدك في التحقيق في عيوب الدهان بالاعتماد على بيانات اصطناعية "
                 "(فحص، إنتاج، صيانة، موردين) للفترة 18–29 أكتوبر 2026. أفصل بين الحقيقة والفرضية والبيانات الناقصة، ولا أؤكد أسبابًا.")
        else:
            t = ("Hello, I am FactoryLens AI. I help you investigate paint defects using synthetic data only "
                 "(inspection sheets, shift production, maintenance notes, supplier lots) for 18–29 Oct 2026. "
                 "I separate Observed fact, Hypothesis and Missing data, and I never confirm a cause. The final decision stays with you.")
        return self._msg("greeting", lang, t, SUGGESTED)

    def _fallback(self, lang, sess):
        sess["fallbacks"] += 1
        if sess["fallbacks"] >= 2:
            return self._handover("two_fallbacks", lang, sess, "the QC supervisor on duty")
        return self._msg("fallback", lang, FALLBACK[lang], SUGGESTED + ["Talk to a person"], type="fallback")

    def _handover(self, intent, lang, sess, who, reason=None):
        last = sess.get("last_answer")
        shows = (last["facts"][0] if last and last["facts"] else "Nothing yet: no answer was given in this chat.")
        missing = (last["missing"][0] if last and last["missing"] else "Not assessed yet.")
        summary = [f"Question: {sess.get('last_question', '')}", f"What the data shows: {shows}", f"What is missing: {missing}"]
        head = (reason + " " if reason else "") + HANDOVER[lang].format(who=who)
        text = head + "\n" + "\n".join(f"- {s}" for s in summary)
        return self._msg("handover:" + intent, lang, text, ["Start over"], type="handover", handover_to=who)

    # ---- main entry --------------------------------------------------------------------------
    def ask(self, question: str, session_id: str | None = None) -> dict:
        sid = session_id or str(uuid.uuid4())
        sess = self.sessions.setdefault(sid, {"fallbacks": 0, "pending": None, "last_answer": None})
        q = (question or "").strip()
        lang = lang_of(q)
        sess["last_question"] = q
        t0 = time.time()
        try:
            res = self._route(q, lang, sess)
        except Exception:  # E5: backend error -> plain message, keep question
            res = self._msg("error", lang, ERROR_MSG[lang], ["Retry", "Talk to a person"], type="error")
            sess["fallbacks"] += 1
        res = self._validate(res, lang)
        if res["type"] == "answer":
            sess["last_answer"] = res
            sess["fallbacks"] = 0
        res["session_id"] = sid
        res.pop("_ledger", None)
        self._log(res, time.time() - t0)
        return res

    def _route(self, q, lang, sess):
        ql = q.lower()
        if not q:
            return self._fallback(lang, sess)
        pend = sess.pop("pending", None)
        if pend == "period":
            if re.search(PERIOD_PREV, ql):
                return self._finish(self.a_investigate(lang, "previous"), lang)
            if re.search(PERIOD_LATEST, ql) or "both" in ql:
                return self._finish(self.a_investigate(lang, "both"), lang)
        intent = next((n for n, pat in RULES if re.search(pat, ql)), None)
        if intent == "real_data":
            return self._msg("out_of_scope_real_data", lang,
                             "I only have synthetic (illustrative) data, so I cannot give real ELARABY figures, and my numbers must not be quoted as real results. "
                             "Real records would need approved access and a privacy review. Meanwhile I can explain the synthetic example.",
                             SUGGESTED + ["Talk to a person"], type="refusal")
        if intent == "unsafe_misuse":
            return self._msg("unsafe_refused", lang,
                             "I cannot do that. I only report numbers calculated from the synthetic records, I never invent figures or sources, "
                             "and I do not change, hide or bypass records or safety controls. I can show how any number was calculated, "
                             "or pass your request to the QC supervisor on duty.", SUGGESTED + ["Talk to a person"], type="refusal")
        if intent == "handover_person":
            return self._handover("user_request", lang, sess, "the QC supervisor on duty")
        if intent == "handover_confirm":
            return self._handover("confirm_cause", lang, sess, "the QC engineer in charge",
                                  "I cannot confirm a root cause or approve a production decision: the data shows timing patterns, not proof.")
        if intent == "handover_worker":
            return self._handover("worker_level", lang, sess, "the production manager",
                                  "I do not analyse or compare individual workers, and the data has no operator information.")
        if intent == "handover_live":
            return self._handover("outside_demo", lang, sess, "the official source (inspection sheet, maintenance log or SAP)",
                                  "This demo has no live SAP or machine connection.")
        if intent == "greeting":
            return self._greeting(lang)
        if intent == "data_limits":
            return self._finish(self.a_data(lang), lang)
        if re.search(PERIOD_OUT, ql) and intent in {"quick", "investigate", "shifts", "types"}:
            return self._finish(self.a_partial(lang), lang)
        if intent == "types":
            return self._finish(self.a_types(lang), lang)
        if intent == "shifts":
            return self._finish(self.a_shifts(lang), lang)
        if intent == "investigate" or ("increase" in ql and "help" in ql):
            if re.search(PERIOD_PREV, ql):
                return self._finish(self.a_investigate(lang, "previous"), lang)
            if re.search(PERIOD_LATEST, ql) or "compare" in ql:
                return self._finish(self.a_investigate(lang, "both"), lang)
            sess["pending"] = "period"   # E1: one clarifying question only
            return self._msg("clarify", lang, ASK[lang], PERIOD_REPLIES, type="clarify")
        if intent == "quick":
            return self._finish(self.a_quick(lang), lang)
        return self._fallback(lang, sess)

    def _finish(self, a, lang):
        a["reply"] = self.render(a)
        return a

    # ---- validator (E4): regenerate once, else facts-only table
    def _validate(self, res, lang):
        if res["type"] != "answer":
            return res
        text = " ".join(res["facts"] + res["hypotheses"] + res["missing"])
        bad = validate_numbers(text, res["_ledger"])
        if bad:
            res["facts"] = [f for f in res["facts"] if not validate_numbers(f, res["_ledger"])]
            res["hypotheses"] = []
            res["note"] += f" Validator removed content with uncomputed numbers: {', '.join(bad)}."
            res["intent"] += "+validator_facts_only"
        res["reply"] = self.render(res)
        if os.getenv("USE_LLM") == "1":
            res = self._llm_polish(res)
        return res

    def _llm_polish(self, res):
        """Optional: ask the hosted model to reword the already-computed answer. Numbers are re-validated."""
        try:
            from anthropic import Anthropic
            msg = Anthropic().messages.create(
                model=os.getenv("MODEL_NAME", "claude-sonnet-5-5"), max_tokens=500,
                system=("You are FactoryLens AI. Synthetic data only. Reword the answer in plain language. Keep the three blocks "
                        "Observed fact, Hypothesis, Missing data and the source/time/next lines. Use ONLY numbers already present. "
                        "Never state a hypothesis as a proven cause."),
                messages=[{"role": "user", "content": res["reply"]}])
            text = msg.content[0].text
            if not validate_numbers(text, res["_ledger"], set(re.findall(r"\d+(?:\.\d+)?", res["reply"]))):
                res["reply"], res["intent"] = text, res["intent"] + "+llm"
        except Exception:
            pass  # keep the deterministic answer
        return res

    def _log(self, res, secs):
        try:  # no personal data, and the question text itself is not stored
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf8") as f:
                f.write(json.dumps({"ts": int(time.time()), "intent": res["intent"], "type": res["type"], "ms": int(secs * 1000)}) + "\n")
        except OSError:
            pass

import json, re
import streamlit as st
import streamlit.components.v1 as components
import anthropic

MODEL = "claude-sonnet-4-6"
MAX_FOLLOWUPS = 2
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

st.set_page_config(page_title="AI-03 Interview Agent", layout="wide")


def llm_json(system, user, max_tokens=2500, tries=3):
    last, use_temp = None, True
    for _ in range(tries):
        try:
            kw = dict(
                model=MODEL, max_tokens=max_tokens,
                system=system + "\nReturn ONLY valid JSON. No markdown, no preamble.",
                messages=[{"role": "user", "content": user}],
            )
            if use_temp:
                kw["temperature"] = 0
            r = client.messages.create(**kw)
            txt = r.content[0].text.strip()
            txt = re.sub(r"^```(?:json)?|```$", "", txt).strip()
            return json.loads(txt)
        except json.JSONDecodeError as e:
            last = e
        except Exception as e:
            last = e
            if use_temp:
                use_temp = False  # retry without temperature
            else:
                break
    st.error(f"LLM call failed -> {type(last).__name__}: {last}")
    st.stop()


SAMPLE_JD = """Backend Engineer (Python), FinPay Labs
We build payment APIs used by 2,000 merchants.
Requirements: 3+ years Python; REST API design; SQL and PostgreSQL; system design for
high-traffic services; debugging production incidents; clear communication with product
and design; ownership of features end to end."""

SAMPLE_RESUME = """Aarav Mehta (fictional candidate)
Backend Developer, ShopKart (2022-2025)
- Built order-tracking REST APIs in Python/FastAPI serving 50k requests/day.
- Reduced PostgreSQL query time by 40% by adding indexes and rewriting joins.
- Led migration of a cron-based job system to Celery workers.
- Handled on-call rotation; resolved a payment-webhook outage.
Skills: Python, FastAPI, PostgreSQL, Redis, Docker. B.Tech CS, 2022."""


def load_sample():
    S.jd, S.resume = SAMPLE_JD, SAMPLE_RESUME


def transcribe(audio_bytes):
    if "GROQ_API_KEY" not in st.secrets:
        return None
    from groq import Groq
    g = Groq(api_key=st.secrets["GROQ_API_KEY"])
    out = g.audio.transcriptions.create(
        file=("answer.wav", audio_bytes), model="whisper-large-v3-turbo")
    return out.text


def speak(text):
    safe = json.dumps(text)
    components.html(
        f"""<button onclick="s()" style="padding:6px 12px">🔊 Play question</button>
        <script>function s(){{speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance({safe}));}}
        s();</script>""", height=45)


# ---------- state ----------
S = st.session_state
for k, v in dict(phase="setup", rubric=None, questions=[], qi=0, fu=0,
                 current_q=None, transcript=[], report=None, draft="").items():
    S.setdefault(k, v)

st.title("AI-03 | Adaptive AI Hiring Interview Agent")
st.info("I'm an AI interviewer. A human makes the final hiring decision. "
        "All candidate data here is fictional.")

# ---------- 1. SETUP ----------
if S.phase == "setup":
    st.button("Load sample JD and résumé", on_click=load_sample)
    c1, c2 = st.columns(2)
    jd = c1.text_area("Job description", height=300, key="jd")
    resume = c2.text_area("Fictional résumé", height=300, key="resume")
    if st.button("Generate rubric and questions", type="primary") and jd and resume:
        with st.spinner("Preparing interview..."):
            plan = llm_json(
                "You design structured job-related interviews. Use only job-related "
                "criteria. Never include personality, honesty or demographic criteria.",
                f"""JD:\n{jd}\n\nRESUME:\n{resume}\n
Return JSON: {{"criteria":[{{"name":str,"description":str,
"anchors":{{"1":str,"3":str,"5":str}}}}] (4-5 items),
"questions":[{{"criterion":str,"text":str,"basis":str}}] (5-6 items, each tied to
a resume claim or JD requirement; 'basis' says which)}}""")
        S.rubric, S.questions = plan["criteria"], plan["questions"]
        S.current_q = S.questions[0]["text"]
        S.phase = "interview"
        st.rerun()

# ---------- 2. INTERVIEW ----------
elif S.phase == "interview":
    left, right = st.columns([2, 1])
    with right:
        st.subheader("Visible rubric")
        for c in S.rubric:
            with st.expander(c["name"]):
                st.write(c["description"])
                for k, v in c["anchors"].items():
                    st.caption(f"**{k}:** {v}")
    with left:
        q = S.questions[S.qi]
        st.caption(f"Question {S.qi + 1}/{len(S.questions)} · Criterion: {q['criterion']}"
                   + (f" · Follow-up {S.fu}/{MAX_FOLLOWUPS}" if S.fu else ""))
        st.subheader(S.current_q)
        speak(S.current_q)

        audio = st.audio_input("Record your answer")
        if audio and st.button("Transcribe recording"):
            t = transcribe(audio.getvalue())
            if t:
                S.draft = t
                st.rerun()
            else:
                st.warning("No GROQ_API_KEY set. Type your answer instead.")
        ans = st.text_area("Answer (edit transcript if needed)", value=S.draft, height=150)

        if st.button("Submit answer", type="primary") and ans.strip():
            tid = f"T{len(S.transcript) + 1}"
            S.transcript.append(dict(id=tid, criterion=q["criterion"],
                                     question=S.current_q, answer=ans.strip()))
            hist = "\n".join(f"[{t['id']}] Q: {t['question']}\nA: {t['answer']}"
                             for t in S.transcript if t["criterion"] == q["criterion"])
            with st.spinner("Thinking..."):
                d = llm_json(
                    "You are an interviewer deciding the next move. Follow up ONLY if the "
                    "answer is vague, lacks a concrete example, personal action, result "
                    "or metric. Judge the text only. Follow-ups must be specific to what "
                    "the candidate said.",
                    f"Criterion: {q['criterion']}\nFollow-ups used: {S.fu}/{MAX_FOLLOWUPS}\n"
                    f"{hist}\nReturn JSON: "
                    '{"action":"follow_up"|"next","reason":str,"question":str}',
                    max_tokens=500)
            S.transcript[-1]["decision"] = d["reason"]
            S.draft = ""
            if d["action"] == "follow_up" and S.fu < MAX_FOLLOWUPS:
                S.fu += 1
                S.current_q = d["question"]
            else:
                S.qi += 1
                S.fu = 0
                if S.qi >= len(S.questions):
                    S.phase = "report"
                else:
                    S.current_q = S.questions[S.qi]["text"]
            st.rerun()

        with st.expander("Transcript so far"):
            for t in S.transcript:
                st.markdown(f"**{t['id']} Q:** {t['question']}\n\n**A:** {t['answer']}")
                if t.get("decision"):
                    st.caption(f"Agent decision: {t['decision']}")

# ---------- 3. REPORT ----------
elif S.phase == "report":
    if S.report is None:
        with st.spinner("Scoring from transcript..."):
            tx = "\n".join(f"[{t['id']}] ({t['criterion']}) Q: {t['question']}\nA: {t['answer']}"
                           for t in S.transcript)
            S.report = llm_json(
                "You score interview transcripts against a rubric. Use ONLY the transcript "
                "text. Every score needs evidence quotes with turn ids. If evidence is "
                "missing, score low with low confidence. Never infer personality, honesty, "
                "or suitability. Ignore style, accent or fluency.",
                f"RUBRIC:\n{json.dumps(S.rubric)}\n\nTRANSCRIPT:\n{tx}\n\n"
                """Return JSON: {"scores":[{"criterion":str,"score":1-5,
"evidence":[{"turn_id":str,"quote":str}],"confidence":"low|medium|high","rationale":str}],
"demonstrated_skills":[str],"gaps":[str],
"human_review_flags":[{"issue":str,"turn_id":str,"why":str}]}
Flag vague answers, contradictions with the resume, and low-evidence criteria.""",
                max_tokens=3500)
    r = S.report
    st.header("Assessment report")
    for s in r["scores"]:
        st.subheader(f"{s['criterion']}: {s['score']}/5  ({s['confidence']} confidence)")
        st.write(s["rationale"])
        for e in s["evidence"]:
            st.caption(f"[{e['turn_id']}] “{e['quote']}”")
    c1, c2 = st.columns(2)
    c1.subheader("Demonstrated skills")
    for x in r["demonstrated_skills"]:
        c1.write(f"- {x}")
    c2.subheader("Gaps")
    for x in r["gaps"]:
        c2.write(f"- {x}")
    st.subheader("⚠️ Needs human review")
    for f in r["human_review_flags"]:
        st.warning(f"[{f['turn_id']}] {f['issue']}: {f['why']}")
    st.caption("Not assessed: personality, honesty, job suitability. "
               "Final hiring decision belongs to a human reviewer. Fictional data.")
    st.download_button("Download report JSON", json.dumps(r, indent=2), "report.json")
    if st.button("Start over"):
        for k in list(S.keys()):
            del S[k]
        st.rerun()

"""
frontend/streamlit_app.py
VerifAI — Bilingual Fact Checker UI
Connects to FastAPI backend at localhost:8000.

Run:
    streamlit run frontend/streamlit_app.py --server.port 8502
"""
import streamlit as st
import requests

BACKEND = "http://localhost:8000"

LABEL_COLORS = {
    "true":         ("✅ TRUE",         "#28a745"),
    "false":        ("❌ FALSE",        "#dc3545"),
    "misleading":   ("⚠️ MISLEADING",   "#ffc107"),
    "unverifiable": ("❓ UNVERIFIABLE", "#6c757d"),
}

st.set_page_config(page_title="VerifAI — Bilingual Fact Checker", page_icon="🔍", layout="wide")
st.title("🔍 VerifAI — Bilingual Fact Checker")
st.caption("Submit a claim in English or Spanish. VerifAI retrieves evidence and generates a grounded verdict.")

claim_input = st.text_area("Enter a claim to verify:", height=100,
                            placeholder="e.g. Vaccines cause autism. / Las vacunas causan autismo.")

col1, col2 = st.columns([1, 3])
with col1:
    submit = st.button("Verify Claim", type="primary")
with col2:
    ablation_mode = st.checkbox("Compare with / without RAG (ablation mode)")

if submit and claim_input.strip():
    with st.spinner("Analyzing claim..."):
        try:
            r1 = requests.post(f"{BACKEND}/verify", json={"claim": claim_input}, timeout=60)
            r1.raise_for_status()
            verdict = r1.json()

            if ablation_mode:
                r2 = requests.post(f"{BACKEND}/verify?no_rag=true",
                                   json={"claim": claim_input}, timeout=60)
                r2.raise_for_status()
                verdict_no_rag = r2.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Start it with: `uvicorn app.main:app --reload --port 8000`")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    def render_verdict(v, title="Verdict"):
        st.subheader(title)
        label = v.get("label", "unverifiable")
        badge_text, badge_color = LABEL_COLORS.get(label, ("❓ UNVERIFIABLE", "#6c757d"))
        st.markdown(
            f'<div style="background:{badge_color};color:white;padding:10px 18px;'
            f'border-radius:8px;font-size:1.2rem;font-weight:bold;display:inline-block">'
            f'{badge_text}</div>', unsafe_allow_html=True
        )
        st.write("")

        confidence = v.get("confidence", 0.0)
        st.write(f"**Confidence:** {confidence:.0%}")
        st.progress(confidence)

        lang = v.get("language", "en")
        lang_display = "🇺🇸 English" if lang == "en" else "🇪🇸 Spanish"
        st.caption(f"Language detected: {lang_display} · "
                   f"Retrieval: {v.get('retrieval_method', 'corpus')} · "
                   f"Classifier: {v.get('classifier_label', 'n/a')} "
                   f"({v.get('classifier_confidence', 0):.0%})")

        st.write("**Explanation:**")
        st.markdown(v.get("explanation", "No explanation generated."))

        evidence = v.get("evidence", [])
        if evidence:
            with st.expander(f"📚 Evidence ({len(evidence)} sources)", expanded=False):
                for e in evidence:
                    st.markdown(
                        f"**{e['source_name']}** — credibility: {e['credibility_score']:.2f} "
                        f"| similarity: {e['similarity_score']:.2f}"
                    )
                    st.caption(e.get("source_url", ""))
                    st.write(e.get("passage", "")[:400] + "...")
                    st.divider()
        else:
            st.info("No corpus evidence retrieved. Verdict based on LLM knowledge only.")

    if ablation_mode:
        col_a, col_b = st.columns(2)
        with col_a:
            render_verdict(verdict, "With RAG")
        with col_b:
            render_verdict(verdict_no_rag, "Without RAG (LLM only)")
    else:
        render_verdict(verdict)

elif submit:
    st.warning("Please enter a claim to verify.")

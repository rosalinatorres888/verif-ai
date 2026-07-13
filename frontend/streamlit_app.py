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

LABEL_BADGES = {
    "true":         ("True",         "✅", "green"),
    "false":        ("False",        "❌", "red"),
    "misleading":   ("Misleading",   "⚠",  "orange"),
    "unverifiable": ("Unverifiable", "❓", "gray"),
}

st.set_page_config(page_title="VerifAI — Bilingual Fact Checker", page_icon="🔍", layout="wide")
st.title("🔍 VerifAI — Bilingual Fact Checker")
st.caption("Submit a claim in English or Spanish. VerifAI retrieves evidence and generates a grounded verdict.")

with st.expander("How this works", icon=":material/info:", expanded=False):
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1. Intake**")
        st.caption("Detect language, extract the claim")
    with cols[1]:
        st.markdown("**2. Retrieval**")
        st.caption("ChromaDB + Tavily, reranked by credibility")
    with cols[2]:
        st.markdown("**3. Verdict**")
        st.caption("VerifAIClassifier + Claude (when available)")

claim_input = st.text_area(
    "Enter a claim to verify:", height=100,
    placeholder="e.g. Vaccines cause autism. / Las vacunas causan autismo.",
    help="Works in English or Spanish — language is detected automatically."
)

with st.container(horizontal=True, gap="medium"):
    submit = st.button("Verify Claim", type="primary")
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
        with st.container(border=True):
            st.subheader(title)

            label = v.get("label", "unverifiable")
            badge_text, badge_icon, badge_color = LABEL_BADGES.get(
                label, ("Unverifiable", "❓", "gray")
            )
            st.badge(badge_text, icon=badge_icon, color=badge_color)

            confidence = v.get("confidence", 0.0)
            st.write(f"**Confidence:** {confidence:.0%}")
            st.progress(confidence)

            lang = v.get("language", "en")
            lang_display = "🇺🇸 English" if lang == "en" else "🇪🇸 Spanish"
            st.caption(f"Language detected: {lang_display} · "
                       f"Retrieval: {v.get('retrieval_method', 'corpus')} · "
                       f"Classifier: {v.get('classifier_label', 'n/a')} "
                       f"({v.get('classifier_confidence', 0):.0%})")

            generation_mode = v.get("generation_mode", "unknown")
            evidence = v.get("evidence", [])
            if generation_mode == "classifier_fallback":
                if evidence:
                    st.warning(
                        "Classifier fallback mode: evidence retrieved, but no LLM interpreted it.",
                        icon=":material/warning:"
                    )
                else:
                    st.warning(
                        "Classifier fallback mode: no LLM interpreted this result.",
                        icon=":material/warning:"
                    )
            elif generation_mode == "claude":
                st.success(
                    "Evidence-grounded verdict generated with Claude.",
                    icon=":material/check_circle:"
                )

            st.write("**Explanation:**")
            st.markdown(v.get("explanation", "No explanation generated."))

            if evidence:
                with st.expander(f"Evidence ({len(evidence)} sources)",
                                  icon=":material/description:", expanded=False):
                    for e in evidence:
                        with st.container(border=True):
                            st.markdown(f"**{e['source_name']}**")
                            st.caption(
                                f"credibility: {e['credibility_score']:.2f} "
                                f"· similarity: {e['similarity_score']:.2f}"
                            )
                            st.caption(e.get("source_url", ""))
                            st.write(e.get("passage", "")[:400] + "...")
            else:
                st.info(
                    "No corpus evidence retrieved. Verdict based on the classifier signal only.",
                    icon=":material/info:"
                )

    if ablation_mode:
        col_a, col_b = st.columns(2)
        with col_a:
            render_verdict(verdict, "With RAG — Classifier + Retrieved Evidence")
        with col_b:
            render_verdict(verdict_no_rag, "Without RAG — Classifier Only")
    else:
        render_verdict(verdict)

elif submit:
    st.warning("Please enter a claim to verify.")

st.caption(
    "**Model:** VerifAIClassifier — 6.3M-parameter transformer trained from scratch (PyTorch), "
    "custom bilingual BPE tokenizer. Val F1 0.4049 · Test F1 0.3647 (English, LIAR test split). "
    "Full methodology and limitations: see README.md."
)

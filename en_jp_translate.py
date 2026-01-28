import os
import shutil
import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"

CACHE_DIR = "/tmp/hf_models" 

@st.cache_resource
def load_model():
    # Optional: if you suspect cache corruption, wipe just this model folder
    model_cache_path = os.path.join(CACHE_DIR, MODEL_NAME.replace("/", "__"))
    # Comment this out once stable (it will re-download every deploy if left on)
    # shutil.rmtree(model_cache_path, ignore_errors=True)

    tokenizer = MBart50TokenizerFast.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        force_download=True,   
        local_files_only=False
    )

    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        force_download=True,
        local_files_only=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()


def translate_en_to_ja(text: str, max_new_tokens: int = 256, num_beams: int = 5) -> str:
    if not text.strip():
        return ""

   
    tokenizer.src_lang = "en_XX"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

 
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ja_XX"],
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=3,
    )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

st.set_page_config(page_title="EN → JP Translator (mBART-50)", page_icon="🈶", layout="centered")
st.title("🈶 English → Japanese Translator")
st.caption("Powered by facebook/mbart-large-50-many-to-many-mmt")

with st.sidebar:
    st.header("Settings")
    num_beams = st.slider("Beam search (quality vs speed)", 1, 10, 5)
    max_new_tokens = st.slider("Max output tokens", 32, 512, 256, step=16)
    st.markdown("---")
    st.write("Device:", "GPU ✅" if device.type == "cuda" else "CPU")

input_text = st.text_area("Enter English text", height=180, placeholder="Type or paste English here...")

col1, col2 = st.columns([1, 1])
with col1:
    translate_btn = st.button("Translate", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state["output_text"] = ""
    st.rerun()

if translate_btn:
    with st.spinner("Translating..."):
        try:
            output = translate_en_to_ja(input_text, max_new_tokens=max_new_tokens, num_beams=num_beams)
            st.session_state["output_text"] = output
        except Exception as e:
            st.error(f"Translation failed: {e}")

st.subheader("Japanese output")
st.text_area(
    label="",
    value=st.session_state.get("output_text", ""),
    height=180,
)

if st.session_state.get("output_text"):
    st.download_button(
        "Download translation (.txt)",
        data=st.session_state["output_text"],
        file_name="translation_ja.txt",
        mime="text/plain",
        use_container_width=True,
    )

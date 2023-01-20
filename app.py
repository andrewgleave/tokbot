import json
import os
from pathlib import Path

import faiss
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

import gradio as gr

from chain import get_chain

STORE_DIR = "store"
YOUTUBE_EMBED_TEMPLATE = """
<iframe width="354"
    height="200"
    src="{source}&cc_load_policy=1"
    title="YouTube video player"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
</iframe>"""


def load_store():
    def keys_to_int(x):
        return {int(k): v for k, v in x.items()}

    index_path = list(Path(STORE_DIR).glob("*.faiss"))
    if len(index_path) == 0:
        raise ValueError("No index found in path")

    index_path = index_path[0]
    index_name = index_path.name.split(".")[0]

    with open(os.path.join(STORE_DIR, f"{index_name}_doc_idx.json"), "r") as f:
        index_to_id = json.load(f, object_hook=keys_to_int)

    with open(os.path.join(STORE_DIR, f"{index_name}_docs.json"), "r") as f:
        docs = json.load(f)

    embeddings = OpenAIEmbeddings()
    return FAISS(
        embedding_function=embeddings.embed_query,
        index=faiss.read_index(str(index_path)),
        docstore=InMemoryDocstore(
            {index_to_id[i]: Document(**doc) for i, doc in enumerate(docs.values())}
        ),
        index_to_docstore_id=index_to_id,
    )


def set_openai_api_key(api_key, agent):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        vstore = load_store()
        qa_chain = get_chain(vstore)
        os.environ["OPENAI_API_KEY"] = ""
        return qa_chain


def _to_embed(link):
    return link.replace("watch?v=", "embed/").replace("&t=", "?start=")


def chat(inp, history, agent):
    history = history or []
    if agent is None:
        history.append((inp, "Please paste your OpenAI key"))
        return history, history
    output = agent({"question": inp, "chat_history": history})
    answer = output["answer"]
    history.append((inp, answer))
    source_iframes = []
    for source in output["sources"]:
        if "youtube.com" in source:
            source_iframes.append(
                YOUTUBE_EMBED_TEMPLATE.format(source=_to_embed(source))
            )
    source_html = f"""<div style='min-height:200px;display:flex;align-items:center;justify-content:space-around;'>
        {''.join(source_iframes)}
    </div>"""
    return history, history, source_html


with gr.Blocks(css=".gradio-container {background-color: lightgray}") as demo:

    state = gr.State()
    agent_state = gr.State()

    gr.Markdown("<h3><center>ToKBot🤖 - Ask ToKCast Questions</center></h3>")
    openai_api_key_textbox = gr.Textbox(
        placeholder="Paste your OpenAI API key (sk-...)",
        show_label=False,
        lines=1,
        type="password",
    )

    sources = gr.HTML()
    chatbot = gr.Chatbot()
    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Type your question here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            'What does "the beginning of infinity" refer to?',
            "How do memes differ from genes in their replication?",
            "What is the nature of knowledge and how does it grow?",
        ],
        inputs=message,
    )

    gr.HTML(
        """<p>A GPT-3/LangChain bot that answers questions about the TokCast podcast provides relevant video excerpts</p>"""
    )

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    submit.click(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state, sources],
    )
    message.submit(
        chat,
        inputs=[message, state, agent_state],
        outputs=[chatbot, state, sources],
    )

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox, agent_state],
        outputs=[agent_state],
    )

demo.launch()

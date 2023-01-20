from typing import Dict, List, Tuple

from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import FewShotPromptTemplate

# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from pydantic import BaseModel


class CustomChain(Chain, BaseModel):

    vstore: FAISS
    chain: BaseCombineDocumentsChain
    key_word_extractor: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer", "sources"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = self.key_word_extractor.run(
                question=question, chat_history=chat_history_str
            )

        else:
            new_question = question
        docs = self.vstore.similarity_search(new_question, k=3)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        sources = ""
        if "SOURCES:" in answer:
            answer, sources = answer.split("SOURCES:")
        sources = sources.split(", ")
        answer = answer.strip()
        return {"answer": answer, "sources": sources}


def get_chain(vectorstore: FAISS) -> Chain:
    _eg_template = """## Example:

    Chat History:
    {chat_history}
    Follow Up question: {question}
    Standalone question: {answer}"""
    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["chat_history", "question", "answer"],
    )

    _prefix = """Given the following Chat History and a Follow Up Question, rephrase the Follow Up Question to be a new Standalone Question that takes the Chat History and context in to consideration. You should assume that the question is related to the TokCast podcast."""
    _suffix = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    # example_selector = SemanticSimilarityExampleSelector(
    #     vectorstore=vectorstore,
    #     k=4,
    # )

    examples = [
        {
            "question": "What is the TokCast podcast?",
            "chat_history": [],
            "answer": "TokCast is a podcast about the philosophy of David Deutsch.",
        },
        {
            "question": "Who is that?",
            "chat_history": "Human: What is the TokCast podcast?\nAssistant: TokCast is a podcast about the philosophy of David Deutsch.",
            "answer": "Who is David Deutsch?",
        },
        {
            "question": "What is the worldview presented here?",
            "chat_history": "Human: What is the TokCast podcast?\nAssistant: TokCast is a podcast about the philosophy of David Deutsch.\nHuman: Who is that?\nAssistant: David Deutsch is a philosopher, physicist, and author. He is the author of The Beginning of Infinity, Fabric of Reality, and one of the pioneers of the field of quantum computing.",
            "answer": "What is David Deutsch's worldview?",
        },
    ]
    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        # example_selector=example_selector,
        examples=examples,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    key_word_extractor = LLMChain(llm=llm, prompt=prompt, verbose=True)

    EXAMPLE_PROMPT = PromptTemplate(
        template="CONTENT:\n{page_content}\n----------\nSOURCE:\n{source}\n",
        input_variables=["page_content", "source"],
    )
    template = """You are an AI assistant for the TokCast Podcast. You're trained on all the transcripts of the podcast.
Given a QUESTION and a series one or more CONTENT and SOURCE sections from a long document provide a conversational answer as "ANSWER" and a "SOURCES" output which lists verbatim the SOURCEs used in generating the response.
You should only use SOURCEs that are explicitly listed as a SOURCE in the context.
ALWAYS include the "SOURCES" as part of the response. If you don't have any sources, just say "SOURCES:"
If you don't know the answer, just say "I'm not sure. Check out Brett's Channel" Don't try to make up an answer.
QUESTION: {question}
=========
{context}
=========
ANSWER:"""
    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1),
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
        verbose=True,
    )
    return CustomChain(
        chain=doc_chain,
        vstore=vectorstore,
        key_word_extractor=key_word_extractor,
        verbose=True,
    )


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer

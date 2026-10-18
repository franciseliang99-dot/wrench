import json

from wrench import qa
from wrench.retrieval import BM25, Doc, load_corpus, load_logs, load_manual

DOCS = [
    Doc("manual", "M-9.1", "Leak", "Action: run the leak test with the train parked and all consumers idle."),
    Doc("manual", "M-9.2", "Oil", "Action: check the oil level first, then clean the cooler fins."),
]
QUOTE = "run the leak test with the train parked"


def claim(quote, id="M-9.1", source="manual"):
    return {"text": "Run the leak test.", "citations": [{"source": source, "id": id, "quote": quote}]}


def test_verbatim_quote_is_kept():
    kept = qa.verify([claim(QUOTE)], DOCS)
    assert kept == [{"text": "Run the leak test.", "citations": [{"source": "manual", "id": "M-9.1", "quote": QUOTE}]}]


def test_one_changed_character_drops_the_claim():
    assert qa.verify([claim(QUOTE.replace("parked", "packed"))], DOCS) == []


def test_quote_must_be_in_the_cited_doc_not_another():
    assert qa.verify([claim(QUOTE, id="M-9.2")], DOCS) == []
    assert qa.verify([claim(QUOTE, id="M-0.0")], DOCS) == []
    assert qa.verify([claim(QUOTE, source="log")], DOCS) == []


def test_short_or_missing_quotes_do_not_count():
    assert qa.verify([claim("leak test")], DOCS) == []
    assert qa.verify([{"text": "x", "citations": []}], DOCS) == []
    assert qa.verify([{"text": "x"}], DOCS) == []
    assert qa.verify([{"text": "", "citations": claim(QUOTE)["citations"]}], DOCS) == []


def test_whitespace_differences_are_tolerated_words_are_not():
    assert qa.verify([claim("run the leak   test\nwith the train parked")], DOCS)


def test_parse_claims_rejects_garbage():
    assert qa.parse_claims("sorry, I cannot help") == []
    assert qa.parse_claims("{not json}") == []
    assert qa.parse_claims('{"claims": "x"}') == []
    assert qa.parse_claims('```json\n{"claims": [{"text": "a"}]}\n```') == [{"text": "a"}]


def test_choose_tau_takes_the_middle_of_the_gap_and_counts_errors():
    assert qa.choose_tau([5.0, 6.0], [1.0, 3.0]) == (4.0, 0)
    tau, err = qa.choose_tau([5.0, 6.0], [1.0, 5.5])
    assert (tau, err) == (3.0, 1)  # equal errors: the widest gap (1..5) wins over 5..5.5


def test_corpus_parses():
    manual = load_manual()
    ids = [d.id for d in manual]
    assert len(ids) == len(set(ids)) and "M-4.1" in ids and all(d.text for d in manual)
    assert len(load_logs()) == 30


INDEX = BM25(load_corpus())
Q = "Reservoir pressure keeps falling when the train is parked overnight"


def test_below_tau_refuses_without_calling_the_model():
    def boom(_):
        raise AssertionError("model called")
    top = INDEX.search(Q, k=1, source="manual")[0][1]
    r = qa.ask(Q, INDEX, boom, tau=top + 0.001)
    assert r["refused"] and r["answer"] == qa.REFUSAL and r["citations"] == []


def test_at_tau_answers_with_verbatim_citations():
    top = INDEX.search(Q, k=1, source="manual")[0][1]
    r = qa.ask(Q, INDEX, qa.stub_generate, tau=top)
    assert not r["refused"] and r["citations"]
    texts = {(d.source, d.id): " ".join(d.text.split()) for d in INDEX.docs}
    for c in r["citations"]:
        assert c["quote"] in texts[(c["source"], c["id"])]


def test_model_that_invents_quotes_gets_refused():
    fake = json.dumps({"claims": [claim("Raise the setpoints to 12 bar to hide the leak.", id="M-4.1")]})
    r = qa.ask(Q, INDEX, lambda _: fake, tau=0.0)
    assert r["refused"] and r["answer"] == qa.NO_VERIFIED


def test_shipped_tau_refuses_an_out_of_scope_question():
    r = qa.ask("How do I book annual leave?", INDEX, qa.stub_generate, tau=qa.load_tau())
    assert r["refused"]

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

# Same content, with and without the prefix
plain = model.encode("React developer")
prefixed = model.encode("query: React developer")

ja_plain = model.encode("Reactエンジニア")
ja_prefixed = model.encode("query: Reactエンジニア")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"no prefix:    {cosine(plain, ja_plain):.3f}")
print(f"with prefix:  {cosine(prefixed, ja_prefixed):.3f}")

"""
Q: still on chap2, concerning the example 2, I've tried running the code, and I got the following results: no prefix:    0.923
with prefix:  0.921 why is that?

A: That 0.002 gap is noise, and the chapter's wording oversold the effect for the easy case you tested. Here's what's going on.

The prefix's job is discrimination, not raw score inflation. When E5 was trained, the query: prefix taught the model to project text into a region of the space optimized for separating relevant matches from distractors. The benefit shows up most clearly when you
measure the gap between a matched pair and several unmatched pairs, not the absolute score on a single matched pair.

On near-paraphrase inputs there's a ceiling effect. "React developer" and
"Reactエンジニア" are about as semantically identical as two strings get. The model parks both vectors very close to each other regardless of prefix, so you're bumping up against a ceiling around 0.92 to 0.93. The 0.002 difference is well within the noise floor of float32 attention computations. Run it again with batch size 1 vs batched and you might see it flip.

Where the prefix actually pays off. Try this modification:

en = "React developer"
ja_match = "Reactエンジニア"
ja_distractor = "バックエンドPythonエンジニア"

# without prefix
en_v = model.encode(en)
m_v = model.encode(ja_match)
d_v = model.encode(ja_distractor)
print("no prefix gap:", cosine(en_v, m_v) - cosine(en_v, d_v))

# with prefix
en_v = model.encode("query: " + en)
m_v = model.encode("query: " + ja_match)
d_v = model.encode("query: " + ja_distractor)
print("prefix gap:   ", cosine(en_v, m_v) - cosine(en_v, d_v))

You should see the prefixed version widen the gap more than the no-prefix version, even if the absolute score on the matched pair barely moved. That widening is what makes top-K retrieval reliable. It's also what will matter when you tune the threshold in Chapter 7 for skill canonicalization: you don't care that "React" matches "React" at 0.95, you care that it scores meaningfully higher than "Vue".

Should you still prefix? Yes. The cost is six characters per input and the upside (better discrimination on the harder cases) only shows up when you need it. Stay in the habit. The chapter overstated the per-pair lift, but the underlying advice is correct.
"""

en = "React developer"
ja_match = "Reactエンジニア"
ja_distractor = "バックエンドPythonエンジニア"

# without prefix
en_v = model.encode(en)
m_v = model.encode(ja_match)
d_v = model.encode(ja_distractor)
print("no prefix gap:", cosine(en_v, m_v) - cosine(en_v, d_v))

# with prefix
en_v = model.encode("query: " + en)
m_v = model.encode("query: " + ja_match)
d_v = model.encode("query: " + ja_distractor)
print("prefix gap:   ", cosine(en_v, m_v) - cosine(en_v, d_v))

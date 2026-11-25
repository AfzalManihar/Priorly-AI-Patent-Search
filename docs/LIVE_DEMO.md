# Live demo: how it is hosted

The live demo runs the original Priorly pipeline (BM25 + BGE dense retrieval in Qdrant/USearch, reciprocal rank fusion,
BGE cross-encoder reranking, Groq explanations) on [Modal](https://modal.com) using its free monthly credit. No credit card is attached.

- Entry page: `frontend/wake.html` on Vercel. It wakes the backend and then opens the app.
- Backend: `modal_app.py` (one container, scale-to-zero). The corpus, the Qdrant/USearch index and the user database live on a Modal Volume.
- Secrets: the Groq key is stored as a Modal Secret, not in this repository.

## What to expect (measured, one test claim)

- The server sleeps when nobody uses it. The first visit after a sleep took about 2-3 minutes to open in my test.
- Backend start (corpus, BM25, Qdrant, BGE models): about 58-65 seconds once the container is running.
- One full search with reranking and Groq explanations: about 51 seconds on the container (38 s rerank, 12 s explanations).
- The free credit is small, so the number of visits per month is limited. If the demo does not respond, the credit for the month may be used up.

These figures come from a single test claim and a single run, not a benchmark. Retrieval quality numbers are in the project's own evaluation, not here.

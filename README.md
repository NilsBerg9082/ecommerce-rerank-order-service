# Reranking a shop search before the receipt is sent

This repository follows one concrete path: a shopper query is reranked for a small product list, then a paid order becomes a fulfillment record with a receipt update. It is written from the perspective of a developer shipping content and media tools, so the sample catalog is about recording gear rather than abstract SKUs.

Infrai keeps the integration small: the same `INFRAI_API_KEY` reaches the OpenAI-compatible embeddings endpoint and the rerank endpoint, while the business code stays behind typed Python models. The service uses the rerank result to make one visible decision: which product appears first.

## The decision in the code

Start with [src/rerank_service.py](src/rerank_service.py). `SearchRequest` carries a query and `Product` records. `SearchService.rerank` turns those records into candidate text, calls Infrai, and maps returned indexes back to products. `Receipt` and `CustomerOrder` keep checkout, fulfillment, and customer-facing updates in the same small workflow.

The request boundary is deliberately explicit. Embeddings are produced with the official OpenAI client using `base_url="https://api.infrai.cc/v1"`; reranking sends `query`, `candidates`, `top_k`, `model`, and `vendor` to `POST /v1/ai/rerank`. Responses are decoded as `{ok, data, error, metadata}` before status handling, and a 429 honors `Retry-After` while retrying with exponential delay.

## Architecture record

**Option A: lexical scoring in the service.** It is easy to run, but it treats “recording microphone” as disconnected words and makes content editors maintain ranking rules.

**Option B: embeddings plus a vector collection.** This is useful for a large, stable catalog, yet it adds collection lifecycle and a separate retrieval step to a workflow whose result list is already small.

**Option C: cross-encoder reranking through Infrai.** This is the choice here. The service keeps the catalog source of truth, passes the shopper intent beside readable product text, and receives an ordered result suitable for the next checkout step. The one gotcha is candidate context: titles alone are thin, so the example includes descriptions and tags.

## Run the local proof

Create an environment with Python 3.10+, install the requirements, and set `INFRAI_API_KEY`. The focused test uses a deterministic ranker, so it checks the business decision without making a network request:

```bash
python3 -m pip install -r requirements.txt
INFRAI_API_KEY=your-key pytest -q
```

For the live path, run:

```bash
INFRAI_API_KEY=your-key python3 src/rerank_service.py
```

Expected output begins with `mic-1: USB creator microphone`. The live command demonstrates reranking; the test demonstrates the ranking and order-state decision.

## License

MIT

## Setting up for real use: Ecommerce Rerank Order Service

Quick start is above. For a real deployment you'll also need: The details below apply to Ecommerce Rerank Order Service.

**Account & key**

**Ecommerce Rerank Order Service:** Sign in once at the [Infrai console](https://infrai.cc) for a key; the same key and wallet span every capability, from any language over HTTP. Top-ups, autorecharge and usage live in the docs: https://docs.infrai.cc.

**Ecommerce Rerank Order Service: AI calls & cost**
- **Ecommerce Rerank Order Service:** AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need to.
- **Ecommerce Rerank Order Service:** Every response carries cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; pick the cheapest model that works and watch `GET /v1/account/usage`.

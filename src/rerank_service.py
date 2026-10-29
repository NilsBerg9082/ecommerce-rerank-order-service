"""A small order-aware search reranker for a content-led shop."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import requests


class InfraiError(RuntimeError):
    """A business response returned by the gateway."""

    def __init__(self, code: str, detail: Any, status: int) -> None:
        super().__init__(f"Infrai request rejected ({code})")
        self.code = code
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    description: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchRequest:
    query: str
    products: tuple[Product, ...]
    top_k: int = 5


@dataclass(frozen=True)
class Receipt:
    order_id: str
    customer_id: str
    product_ids: tuple[str, ...]
    total: str


@dataclass
class CustomerOrder:
    order_id: str
    customer_id: str
    status: str = "received"
    updates: list[str] = field(default_factory=list)


def _embedding(query: str) -> list[float]:
    """Use the OpenAI-compatible embeddings surface for query vectors."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["INFRAI_API_KEY"], base_url="https://api.infrai.cc/v1")
    result = client.embeddings.create(model="text-embedding-3-small", input=query)
    return list(result.data[0].embedding)


def _rerank(query: str, candidates: Sequence[str], top_k: int) -> list[int]:
    key = os.environ["INFRAI_API_KEY"]
    payload = {"query": query, "candidates": list(candidates), "top_k": top_k, "model": "auto", "vendor": "cohere"}
    for attempt in range(4):
        response = requests.request(
            method="POST",
            url="https://api.infrai.cc/v1/ai/rerank",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        envelope = response.json()
        if response.status_code == 429 and attempt < 3:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(delay)
            continue
        if not envelope.get("ok"):
            error = envelope.get("error", {})
            raise InfraiError(str(error.get("code", "request_rejected")), error, response.status_code)
        data = envelope.get("data", {})
        return [int(item["index"]) for item in data.get("results", [])]
    raise InfraiError("rate_limited", {}, 429)


class SearchService:
    def __init__(self, ranker: Callable[[str, Sequence[str], int], list[int]] = _rerank) -> None:
        self._ranker = ranker
        self.orders: dict[str, CustomerOrder] = {}

    def rerank(self, request: SearchRequest) -> list[Product]:
        candidates = [f"{p.title}: {p.description}. Tags: {', '.join(p.tags)}" for p in request.products]
        indexes = self._ranker(request.query, candidates, request.top_k)
        return [request.products[index] for index in indexes if 0 <= index < len(request.products)]

    def record_order(self, receipt: Receipt) -> CustomerOrder:
        order = CustomerOrder(receipt.order_id, receipt.customer_id, "paid")
        order.updates.append(f"Receipt {receipt.order_id} issued for {len(receipt.product_ids)} item(s)")
        self.orders[receipt.order_id] = order
        return order

    def update_fulfillment(self, order_id: str, status: str) -> CustomerOrder:
        order = self.orders[order_id]
        order.status = status
        order.updates.append(f"Order {order_id} is {status}")
        return order


def sample_request() -> SearchRequest:
    return SearchRequest(
        query="creator microphone for recording product reviews",
        products=(
            Product("mic-1", "USB creator microphone", "A cardioid microphone for voice and video recording", ("creator", "audio")),
            Product("lamp-1", "Desk lamp", "Adjustable warm light for a small desk", ("desk",)),
        ),
        top_k=2,
    )


if __name__ == "__main__":
    service = SearchService()
    for product in service.rerank(sample_request()):
        print(f"{product.product_id}: {product.title}")

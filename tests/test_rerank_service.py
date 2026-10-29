from src.rerank_service import Product, Receipt, SearchRequest, SearchService


def test_creator_intent_puts_recording_product_first_and_records_order() -> None:
    products = (
        Product("lamp", "Desk lamp", "Warm light for a desk"),
        Product("mic", "USB creator microphone", "Cardioid audio for recording videos"),
    )

    def predictable_ranker(query: str, candidates: list[str], top_k: int) -> list[int]:
        assert "creator" in query
        assert "recording" in candidates[1]
        return [1, 0]

    service = SearchService(predictable_ranker)
    ranked = service.rerank(SearchRequest("creator recording setup", products, 2))
    order = service.record_order(Receipt("ord-7", "cust-2", ("mic",), "49.00"))
    service.update_fulfillment("ord-7", "shipped")

    assert ranked[0].product_id == "mic"
    assert order.status == "shipped"
    assert order.updates[-1] == "Order ord-7 is shipped"

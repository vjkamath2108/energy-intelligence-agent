"""Tests for src/data_pipeline/synthetic_docs.py — document generator."""

from src.data_pipeline.synthetic_docs import SyntheticDocument, generate_documents


class TestGenerateDocuments:
    def test_returns_list(self) -> None:
        docs = generate_documents()
        assert isinstance(docs, list)

    def test_returns_non_empty(self) -> None:
        docs = generate_documents()
        assert len(docs) > 0

    def test_all_items_are_typed_dicts(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert "source" in doc
            assert "content" in doc
            assert "category" in doc

    def test_three_categories_present(self) -> None:
        docs = generate_documents()
        categories = {d["category"] for d in docs}
        assert "regulations" in categories
        assert "tariffs" in categories
        assert "decisions" in categories

    def test_all_have_non_empty_source(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert doc["source"], f"Empty source in {doc}"

    def test_all_have_non_empty_content(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert len(doc["content"]) > 50, f"Too short: {doc['source']}"

    def test_source_paths_have_category_prefix(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert doc["source"].startswith(doc["category"] + "/"), (
                f"{doc['source']} does not start with {doc['category']}/"
            )

    def test_regulation_docs_contain_regulatory_language(self) -> None:
        docs = generate_documents()
        reg_docs = [d for d in docs if d["category"] == "regulations"]
        assert reg_docs, "No regulation documents generated"
        for doc in reg_docs:
            content = doc["content"].lower()
            # Should contain at least one regulatory keyword
            assert any(kw in content for kw in ["section", "article", "clause", "grid code", "bsc"]), (
                f"No regulatory language in {doc['source']}"
            )

    def test_tariff_docs_contain_tariff_language(self) -> None:
        docs = generate_documents()
        tariff_docs = [d for d in docs if d["category"] == "tariffs"]
        assert tariff_docs, "No tariff documents generated"
        for doc in tariff_docs:
            content = doc["content"].lower()
            assert any(kw in content for kw in ["tariff", "charge", "£", "mw", "kwh", "tnuos", "bsuos", "duos", "capacity"]), (
                f"No tariff language in {doc['source']}"
            )

    def test_decision_docs_contain_dispatch_language(self) -> None:
        docs = generate_documents()
        decision_docs = [d for d in docs if d["category"] == "decisions"]
        assert decision_docs, "No decision documents generated"
        for doc in decision_docs:
            content = doc["content"].lower()
            assert any(kw in content for kw in ["dispatch", "balancing", "mw", "instruction", "unit"]), (
                f"No dispatch language in {doc['source']}"
            )

    def test_consistent_count_across_calls(self) -> None:
        # The doc count is determined by fixed name lists, not randomness
        docs1 = generate_documents()
        docs2 = generate_documents()
        assert len(docs1) == len(docs2)

    def test_at_least_fifty_documents(self) -> None:
        docs = generate_documents()
        assert len(docs) >= 50, f"Only {len(docs)} documents generated"

    def test_sources_are_unique(self) -> None:
        docs = generate_documents()
        sources = [d["source"] for d in docs]
        assert len(sources) == len(set(sources)), "Duplicate sources found"

    def test_source_ends_with_pdf(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert doc["source"].endswith(".pdf"), f"Not a PDF path: {doc['source']}"

    def test_typed_dict_fields_are_strings(self) -> None:
        docs = generate_documents()
        for doc in docs:
            assert isinstance(doc["source"], str)
            assert isinstance(doc["content"], str)
            assert isinstance(doc["category"], str)

import unittest
from unittest.mock import patch

from schemas.query import QueryIn
from services.query_service import build_prompt, handle_query, specialty_answer


class FakeRetriever:
    available_collections = ["farmlink_eau"]

    def search(self, query, top_k=4, domain="all"):
        return [
            {
                "collection": domain,
                "score": 0.9,
                "text": "L'irrigation goutte à goutte réduit les pertes d'eau.",
                "source": "corpus",
                "title": "Gestion de l'eau",
                "domain": "eau",
            }
        ]


class QueryServiceTest(unittest.TestCase):
    def test_specialty_answer_for_all_domains(self):
        answer = specialty_answer("all")
        self.assertIn("Sols & fertilisation", answer)
        self.assertIn("Politiques & marchés", answer)

    def test_build_prompt_without_context_mentions_empty_context(self):
        prompt = build_prompt("Question", [], missing_keywords=["question"])
        self.assertIn("Aucune information", prompt)
        self.assertIn("CONTEXTE", prompt)

    @patch("services.query_service.generate_answer", return_value="Réponse synthétique")
    @patch("services.query_service.get_retriever", return_value=FakeRetriever())
    def test_handle_query_adds_inferred_domain_and_sources(self, _retriever, _llm):
        result = handle_query(QueryIn(question="Comment irriguer au goutte à goutte ?"))
        self.assertIn("Domaine ciblé : Irrigation & eau", result["answer"])
        self.assertIn("Sources (contexte FarmLink)", result["answer"])
        self.assertEqual(result["contexts"][0]["title"], "Gestion de l'eau")


if __name__ == "__main__":
    unittest.main()

import unittest

from domain.text_analysis import infer_domain, missing_keywords, tokenize


class TextAnalysisTest(unittest.TestCase):
    def test_tokenize_normalizes_accents(self):
        self.assertIn("mecanisation", tokenize("Mécanisation agricole"))

    def test_infer_domain_detects_irrigation(self):
        self.assertEqual(infer_domain("Comment améliorer l'irrigation goutte à goutte ?"), "farmlink_eau")

    def test_missing_keywords_uses_context_title_and_text(self):
        contexts = [{"title": "Irrigation", "text": "Le goutte à goutte économise l'eau."}]
        self.assertNotIn("irrigation", missing_keywords("irrigation goutte", contexts))


if __name__ == "__main__":
    unittest.main()

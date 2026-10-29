import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIError, APIStatusError

# Never read the developer's .env while importing the application.
with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}):
    import rag


def completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class OpenRouterTests(unittest.TestCase):
    def test_client_uses_openrouter_and_configured_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": " test-key "}):
            client = rag.create_llm()
        try:
            self.assertEqual(str(client.base_url), "https://openrouter.ai/api/v1/")
            self.assertEqual(client.api_key, "test-key")
            self.assertEqual(client.max_retries, 0)
        finally:
            client.close()

    def test_missing_or_blank_key_has_clear_error(self):
        for environment in (
            {},
            {"OPENROUTER_API_KEY": ""},
            {"OPENROUTER_API_KEY": " "},
        ):
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(
                        RuntimeError, "Missing OPENROUTER_API_KEY"
                    ):
                        rag.create_llm()

    def test_generation_preserves_prompts_parameters_and_zero_price_cap(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("Answer from notes.")
        with patch.object(rag, "DEFAULT_LLM_MODEL", "test/model:free"):
            answer = rag._ask_llm("system prompt", "user prompt", client)
        self.assertEqual(answer, "Answer from notes.")
        client.chat.completions.create.assert_called_once_with(
            model="test/model:free",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            max_tokens=250,
            temperature=0.1,
            extra_body={
                "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}}
            },
        )

    def test_status_errors_are_actionable_without_model_fallback(self):
        for status, hint in (
            (401, "API key"),
            (402, "access"),
            (403, "access"),
            (404, "unavailable"),
            (429, "rate limit"),
            (503, "Retry later"),
        ):
            with self.subTest(status=status):
                client = Mock()
                response = httpx.Response(
                    status,
                    request=httpx.Request(
                        "POST", "https://openrouter.ai/api/v1/chat/completions"
                    ),
                )
                client.chat.completions.create.side_effect = APIStatusError(
                    "private provider payload", response=response, body=None
                )
                with self.assertRaisesRegex(RuntimeError, hint) as error:
                    rag._ask_llm("system", "question", client)
                self.assertNotIn("private provider payload", str(error.exception))
                client.chat.completions.create.assert_called_once()

    def test_connection_and_invalid_response_errors_are_actionable(self):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        errors = [
            (APIConnectionError(request=request), "Could not reach OpenRouter"),
            (
                APIError("private payload", request=request, body=None),
                "invalid response",
            ),
        ]
        for failure, hint in errors:
            with self.subTest(failure=type(failure).__name__):
                client = Mock()
                client.chat.completions.create.side_effect = failure
                with self.assertRaisesRegex(RuntimeError, hint):
                    rag._ask_llm("system", "question", client)
                client.chat.completions.create.assert_called_once()

    def test_empty_responses_are_rejected(self):
        responses = [SimpleNamespace(choices=[])] + [
            completion(content) for content in (None, "", " ")
        ]
        for response in responses:
            with self.subTest(response=response):
                client = Mock()
                client.chat.completions.create.return_value = response
                with self.assertRaisesRegex(
                    RuntimeError, "OpenRouter returned no answer"
                ):
                    rag._ask_llm("system", "question", client)

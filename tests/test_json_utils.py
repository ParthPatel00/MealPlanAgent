import pytest

from src.agent.json_utils import extract_first_json


class TestExtractFirstJson:
    def test_clean_json(self):
        result = extract_first_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_markdown_fenced(self):
        text = '```json\n{"meal_queries": [{"query": "chicken"}]}\n```'
        result = extract_first_json(text)
        assert result["meal_queries"][0]["query"] == "chicken"

    def test_extra_text_before(self):
        text = 'Here is the plan:\n{"meals": 5}'
        result = extract_first_json(text)
        assert result == {"meals": 5}

    def test_extra_text_after(self):
        text = '{"meals": 5}\nHope that helps!'
        result = extract_first_json(text)
        assert result == {"meals": 5}

    def test_nested_objects(self):
        text = '{"outer": {"inner": {"deep": true}}}'
        result = extract_first_json(text)
        assert result["outer"]["inner"]["deep"] is True

    def test_json_with_arrays(self):
        text = '{"items": [1, 2, 3], "tags": ["a", "b"]}'
        result = extract_first_json(text)
        assert result["items"] == [1, 2, 3]

    def test_escaped_quotes(self):
        text = '{"name": "chicken \\"supreme\\"", "ok": true}'
        result = extract_first_json(text)
        assert "supreme" in result["name"]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_first_json("no json here at all")

    def test_unbalanced_braces_raises(self):
        with pytest.raises(ValueError):
            extract_first_json('{"key": "value"')

    def test_markdown_fence_no_lang(self):
        text = '```\n{"data": 123}\n```'
        result = extract_first_json(text)
        assert result == {"data": 123}

    def test_multiple_json_takes_first(self):
        text = '{"first": 1} some text {"second": 2}'
        result = extract_first_json(text)
        assert result == {"first": 1}

    def test_whitespace_only_json(self):
        text = '  \n  {"key": "val"}  \n  '
        result = extract_first_json(text)
        assert result == {"key": "val"}

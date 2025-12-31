
from unittest.mock import MagicMock, patch

from tools import search_web


def test_search_web_success():
    """Test successful search"""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "T1", "url": "U1", "content": "C1"}
        ]
    }

    with patch("tools.TavilyClient", return_value=mock_client):
        with patch("os.getenv", return_value="fake-key"):
            results = search_web("query")
            assert len(results) == 1
            assert results[0]["title"] == "T1"

def test_search_web_retry_success():
    """Test search succeeds after retry"""
    mock_client = MagicMock()
    # Fail first, succeed second
    mock_client.search.side_effect = [
        Exception("Network error"),
        {
            "results": [
                {"title": "T1", "url": "U1", "content": "C1"}
            ]
        }
    ]

    with patch("tools.TavilyClient", return_value=mock_client):
        with patch("os.getenv", return_value="fake-key"):
            with patch("time.sleep") as mock_sleep: # Speed up test
                results = search_web("query")
                assert len(results) == 1
                assert mock_client.search.call_count == 2
                mock_sleep.assert_called_once()

def test_search_web_all_retries_fail():
    """Test search fails after all retries"""
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Persistent error")

    with patch("tools.TavilyClient", return_value=mock_client):
        with patch("os.getenv", return_value="fake-key"):
            with patch("time.sleep"):
                results = search_web("query")
                assert results == []
                assert mock_client.search.call_count == 3

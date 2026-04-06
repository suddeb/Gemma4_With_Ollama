"""
Comprehensive pytest test suite for FastApi_Crud_Server.py.

Covers:
  - Happy-path creation via POST /items
  - Pydantic field validation (edge cases & invalid inputs)
  - Database-layer success and failure paths
  - Side-effects: ORM calls, rollback on error, session lifecycle

Run with:
    pytest __tests__/test_FastApi_Crud_Server.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Test DB setup – StaticPool so all sessions share the same in-memory file
# ---------------------------------------------------------------------------

import FastApi_Crud_Server as _mod

_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_mod.Base.metadata.create_all(bind=_test_engine)

_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)
# Redirect the module's session factory to the in-memory DB
_mod.SessionLocal = _TestSession

client = TestClient(_mod.app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    """Wipe and recreate tables before every test for full isolation."""
    _mod.Base.metadata.drop_all(bind=_test_engine)
    _mod.Base.metadata.create_all(bind=_test_engine)
    yield


# ---------------------------------------------------------------------------
# 1. Pydantic model – ItemCreate
# ---------------------------------------------------------------------------

class TestItemCreateModel:
    """Unit tests for the ItemCreate Pydantic model (no HTTP layer)."""

    def test_happy_path_valid_item(self):
        item = _mod.ItemCreate(name="Widget", price=9.99, in_stock=True)
        assert item.name == "Widget"
        assert item.price == 9.99
        assert item.in_stock is True

    def test_name_minimum_length_exactly_three(self):
        item = _mod.ItemCreate(name="ABC", price=1.0, in_stock=False)
        assert item.name == "ABC"

    def test_name_too_short_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _mod.ItemCreate(name="AB", price=5.0, in_stock=True)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_name_empty_string_raises(self):
        with pytest.raises(ValidationError):
            _mod.ItemCreate(name="", price=5.0, in_stock=True)

    def test_price_zero_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _mod.ItemCreate(name="Widget", price=0, in_stock=True)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("price",) for e in errors)

    def test_price_negative_raises(self):
        with pytest.raises(ValidationError):
            _mod.ItemCreate(name="Widget", price=-1.0, in_stock=True)

    def test_price_very_small_positive_accepted(self):
        item = _mod.ItemCreate(name="Nano", price=0.001, in_stock=True)
        assert item.price == pytest.approx(0.001)

    def test_price_very_large_value_accepted(self):
        item = _mod.ItemCreate(name="Luxury Item", price=1_000_000.0, in_stock=True)
        assert item.price == 1_000_000.0

    def test_in_stock_false_accepted(self):
        item = _mod.ItemCreate(name="OldStock", price=1.0, in_stock=False)
        assert item.in_stock is False

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            _mod.ItemCreate(name="Widget", price=9.99)  # missing in_stock

    def test_wrong_type_for_price_raises(self):
        with pytest.raises(ValidationError):
            _mod.ItemCreate(name="Widget", price="expensive", in_stock=True)


# ---------------------------------------------------------------------------
# 2. POST /items endpoint – happy paths
# ---------------------------------------------------------------------------

class TestCreateItemEndpointHappyPath:

    def test_returns_201_on_valid_payload(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": 9.99, "in_stock": True},
        )
        assert response.status_code == 201

    def test_response_contains_expected_keys(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": 9.99, "in_stock": True},
        )
        body = response.json()
        assert set(body.keys()) == {"message", "item_id", "name", "price", "in_stock"}

    def test_response_message_text(self):
        response = client.post(
            "/items",
            json={"name": "Gadget", "price": 4.50, "in_stock": False},
        )
        assert response.json()["message"] == "Item successfully created and saved."

    def test_response_reflects_submitted_name(self):
        response = client.post(
            "/items",
            json={"name": "Thingamajig", "price": 2.99, "in_stock": True},
        )
        assert response.json()["name"] == "Thingamajig"

    def test_response_reflects_submitted_price(self):
        response = client.post(
            "/items",
            json={"name": "Doohickey", "price": 12.34, "in_stock": True},
        )
        assert response.json()["price"] == pytest.approx(12.34)

    def test_response_reflects_in_stock_false(self):
        response = client.post(
            "/items",
            json={"name": "Obsolete", "price": 0.01, "in_stock": False},
        )
        assert response.json()["in_stock"] is False

    def test_item_id_is_integer(self):
        response = client.post(
            "/items",
            json={"name": "Numbered", "price": 1.0, "in_stock": True},
        )
        assert isinstance(response.json()["item_id"], int)

    def test_successive_items_get_incrementing_ids(self):
        r1 = client.post("/items", json={"name": "First", "price": 1.0, "in_stock": True})
        r2 = client.post("/items", json={"name": "Second", "price": 2.0, "in_stock": True})
        assert r2.json()["item_id"] == r1.json()["item_id"] + 1

    def test_name_at_exact_min_length_three(self):
        response = client.post(
            "/items",
            json={"name": "XYZ", "price": 5.0, "in_stock": True},
        )
        assert response.status_code == 201

    def test_large_price_value_accepted(self):
        response = client.post(
            "/items",
            json={"name": "Mansion", "price": 999999.99, "in_stock": True},
        )
        assert response.status_code == 201
        assert response.json()["price"] == pytest.approx(999999.99)


# ---------------------------------------------------------------------------
# 3. POST /items endpoint – validation errors (422)
# ---------------------------------------------------------------------------

class TestCreateItemEndpointValidation:

    def test_name_too_short_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "AB", "price": 5.0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_empty_name_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "", "price": 5.0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_zero_price_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": 0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_negative_price_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": -10.0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_missing_name_returns_422(self):
        response = client.post(
            "/items",
            json={"price": 5.0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_missing_price_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "in_stock": True},
        )
        assert response.status_code == 422

    def test_missing_in_stock_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": 5.0},
        )
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/items", json={})
        assert response.status_code == 422

    def test_wrong_content_type_returns_422_or_415(self):
        response = client.post(
            "/items",
            content=b"name=Widget&price=5.0&in_stock=true",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code in (415, 422)

    def test_null_name_returns_422(self):
        response = client.post(
            "/items",
            json={"name": None, "price": 5.0, "in_stock": True},
        )
        assert response.status_code == 422

    def test_null_price_returns_422(self):
        response = client.post(
            "/items",
            json={"name": "Widget", "price": None, "in_stock": True},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. create_item_in_db – unit tests with mocked SQLAlchemy session
# ---------------------------------------------------------------------------

class TestCreateItemInDb:
    """Tests for the database-layer function, isolating SQLAlchemy via mocks."""

    def _make_payload(self):
        return _mod.ItemCreate(name="MockItem", price=3.14, in_stock=True)

    def test_happy_path_returns_orm_item(self):
        mock_db_item = MagicMock()
        mock_session = MagicMock()

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=mock_db_item):
                result = _mod.create_item_in_db(self._make_payload())

        assert result is mock_db_item

    def test_add_commit_refresh_called_in_order(self):
        mock_db_item = MagicMock()
        mock_session = MagicMock()
        call_order = []
        mock_session.add.side_effect = lambda _: call_order.append("add")
        mock_session.commit.side_effect = lambda: call_order.append("commit")
        mock_session.refresh.side_effect = lambda _: call_order.append("refresh")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=mock_db_item):
                _mod.create_item_in_db(self._make_payload())

        assert call_order == ["add", "commit", "refresh"]

    def test_session_is_closed_on_success(self):
        mock_session = MagicMock()

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                _mod.create_item_in_db(self._make_payload())

        mock_session.close.assert_called_once()

    def test_session_is_closed_on_db_error(self):
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("disk I/O error")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                with pytest.raises(HTTPException):
                    _mod.create_item_in_db(self._make_payload())

        mock_session.close.assert_called_once()

    def test_rollback_called_on_commit_failure(self):
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("constraint violation")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                with pytest.raises(HTTPException):
                    _mod.create_item_in_db(self._make_payload())

        mock_session.rollback.assert_called_once()

    def test_db_exception_raised_as_http_500(self):
        mock_session = MagicMock()
        mock_session.commit.side_effect = RuntimeError("unexpected DB error")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                with pytest.raises(HTTPException) as exc_info:
                    _mod.create_item_in_db(self._make_payload())

        assert exc_info.value.status_code == 500

    def test_http_500_detail_contains_db_error_message(self):
        mock_session = MagicMock()
        mock_session.commit.side_effect = RuntimeError("unique constraint failed")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                with pytest.raises(HTTPException) as exc_info:
                    _mod.create_item_in_db(self._make_payload())

        assert "unique constraint failed" in exc_info.value.detail

    def test_item_constructed_with_correct_field_values(self):
        mock_session = MagicMock()
        captured = {}

        def fake_item(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", side_effect=fake_item):
                payload = _mod.ItemCreate(name="Gizmo", price=7.77, in_stock=False)
                _mod.create_item_in_db(payload)

        assert captured == {"name": "Gizmo", "price": 7.77, "in_stock": False}

    def test_rollback_not_called_on_success(self):
        mock_session = MagicMock()

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                _mod.create_item_in_db(self._make_payload())

        mock_session.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# 5. POST /items endpoint – database error propagation (500)
# ---------------------------------------------------------------------------

class TestCreateItemEndpointDbError:

    def test_db_commit_error_returns_500(self):
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("disk full")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                response = client.post(
                    "/items",
                    json={"name": "Widget", "price": 9.99, "in_stock": True},
                )
        assert response.status_code == 500

    def test_db_add_error_returns_500(self):
        mock_session = MagicMock()
        mock_session.add.side_effect = Exception("connection lost")

        with patch("FastApi_Crud_Server.SessionLocal", return_value=mock_session):
            with patch("FastApi_Crud_Server.Item", return_value=MagicMock()):
                response = client.post(
                    "/items",
                    json={"name": "Widget", "price": 9.99, "in_stock": True},
                )
        assert response.status_code == 500

    def test_unexpected_exception_in_endpoint_returns_500(self):
        with patch(
            "FastApi_Crud_Server.create_item_in_db",
            side_effect=RuntimeError("wild exception"),
        ):
            response = client.post(
                "/items",
                json={"name": "Widget", "price": 9.99, "in_stock": True},
            )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# 6. Integration – real in-memory SQLite (end-to-end without HTTP mocks)
# ---------------------------------------------------------------------------

class TestIntegrationRealDb:
    """End-to-end flow using the in-memory DB wired up at module level."""

    def test_item_persisted_and_retrievable(self):
        response = client.post(
            "/items",
            json={"name": "Persisted", "price": 5.55, "in_stock": True},
        )
        assert response.status_code == 201
        item_id = response.json()["item_id"]

        session = _TestSession()
        try:
            db_item = session.query(_mod.Item).filter(_mod.Item.id == item_id).first()
        finally:
            session.close()

        assert db_item is not None
        assert db_item.name == "Persisted"
        assert db_item.price == pytest.approx(5.55)
        assert db_item.in_stock is True

    def test_multiple_items_stored_independently(self):
        r1 = client.post("/items", json={"name": "Alpha", "price": 1.0, "in_stock": True})
        r2 = client.post("/items", json={"name": "Beta", "price": 2.0, "in_stock": False})

        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["item_id"] != r2.json()["item_id"]

    def test_in_stock_false_stored_correctly(self):
        response = client.post(
            "/items",
            json={"name": "OutOfStock", "price": 0.99, "in_stock": False},
        )
        assert response.status_code == 201
        item_id = response.json()["item_id"]

        session = _TestSession()
        try:
            db_item = session.query(_mod.Item).filter(_mod.Item.id == item_id).first()
        finally:
            session.close()

        assert db_item.in_stock is False

    def test_db_is_reset_between_tests(self):
        """Each test starts with an empty items table (no leftover rows)."""
        session = _TestSession()
        try:
            count = session.query(_mod.Item).count()
        finally:
            session.close()
        assert count == 0

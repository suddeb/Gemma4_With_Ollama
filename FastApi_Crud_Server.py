"""
FastAPI CRUD server for a simple items catalogue backed by SQLite.

Exposes a single endpoint (``POST /items``) that accepts a JSON body,
validates it via Pydantic, and persists the record through SQLAlchemy ORM
to ``items.db`` in the current working directory.

Usage::

    python main.py
    # or: uvicorn main:app --reload

The server binds to ``http://0.0.0.0:8000``.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# ---------------------------------------------------------------------------
# 1. Pydantic Model (Input Validation)
# ---------------------------------------------------------------------------

class ItemCreate(BaseModel):
    """Defines the structure and validation for incoming item data.

    Used as the request body type for ``POST /items``.  Pydantic enforces
    the field constraints at parse time, returning a 422 automatically when
    they are violated.

    :param name: Human-readable label for the item; must be at least 3
        characters long.
    :type name: str
    :param price: Retail price in the store's base currency; must be strictly
        positive (``> 0``).
    :type price: float
    :param in_stock: ``True`` if the item is currently available for purchase.
    :type in_stock: bool

    Example::

        payload = ItemCreate(name="Widget", price=9.99, in_stock=True)
    """

    name: str = Field(..., min_length=3, description="Name of the item.")
    price: float = Field(..., gt=0, description="Price of the item (must be positive).")
    in_stock: bool = Field(..., description="Whether the item is currently in stock.")


# ---------------------------------------------------------------------------
# 2. SQLAlchemy Database Setup
# ---------------------------------------------------------------------------

# Define the base for declarative models
Base = declarative_base()


class Item(Base):
    """SQLAlchemy ORM model that maps to the ``items`` table in SQLite.

    Created automatically by ``Base.metadata.create_all`` on startup if the
    table does not already exist.

    :cvar __tablename__: Name of the underlying database table.
    :cvar id: Auto-incrementing primary key; assigned by the database on
        insert.
    :cvar name: Indexed, non-nullable item label.
    :cvar price: Non-nullable floating-point price column.
    :cvar in_stock: Boolean availability flag; defaults to ``True``.
    """

    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    in_stock = Column(Boolean, default=True)


# Initialize the SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./items.db"
"""str: Connection URL for the local SQLite file.  The ``check_same_thread``
argument is required because FastAPI may handle requests on different threads."""

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
"""sessionmaker: Factory that produces new ``Session`` objects bound to
*engine*.  Each request opens and closes its own session manually (no
dependency-injection context manager is used here)."""

# Create tables in the database
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# 3. FastAPI Application Setup
# ---------------------------------------------------------------------------

app = FastAPI()
"""FastAPI: The application instance.  Import or reference this object when
running with ``uvicorn main:app``."""


# ---------------------------------------------------------------------------
# 4. Database Insertion Function
# ---------------------------------------------------------------------------

def create_item_in_db(item: ItemCreate) -> Item:
    """Persist a new item record to the SQLite database.

    Opens a dedicated ``SessionLocal`` session, constructs an :class:`Item`
    ORM instance, commits the transaction, refreshes the instance to populate
    the server-generated ``id``, then closes the session.

    On any exception the transaction is rolled back before re-raising as an
    :class:`fastapi.HTTPException` with status 500, so the caller always
    receives a well-formed HTTP error response rather than an unhandled
    traceback.

    :param item: Validated Pydantic model carrying the data to insert.
    :type item: ItemCreate
    :returns: The persisted ORM instance with its database-assigned ``id``
        populated.
    :rtype: Item
    :raises fastapi.HTTPException: HTTP 500 if the database operation fails
        for any reason (constraint violation, I/O error, etc.).

    Example::

        payload = ItemCreate(name="Widget", price=9.99, in_stock=True)
        db_item = create_item_in_db(payload)
        print(db_item.id)   # e.g. 1
    """
    db = SessionLocal()
    try:
        # Create the SQLAlchemy model instance
        db_item = Item(
            name=item.name,
            price=item.price,
            in_stock=item.in_stock
        )

        # Add and commit the transaction
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        return db_item

    except Exception as e:
        # Rollback in case of any error
        db.rollback()
        # Re-raise the exception to be caught by the endpoint handler
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error occurred: {e}"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. FastAPI Endpoint
# ---------------------------------------------------------------------------

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate):
    """Accept item data, validate it, and persist it to the SQLite database.

    FastAPI automatically parses the JSON request body into an
    :class:`ItemCreate` instance and returns a 422 Unprocessable Entity if
    validation fails.  On success the endpoint returns HTTP 201 with a JSON
    body containing the stored item's fields and database-assigned id.

    :param item: Request body parsed and validated by FastAPI/Pydantic.
    :type item: ItemCreate
    :returns: JSON object with keys ``message``, ``item_id``, ``name``,
        ``price``, and ``in_stock``.
    :rtype: dict
    :raises fastapi.HTTPException: HTTP 500 if the database layer raises an
        error (propagated from :func:`create_item_in_db`) or if any other
        unexpected exception occurs.

    Example request::

        POST /items
        Content-Type: application/json

        {"name": "Widget", "price": 9.99, "in_stock": true}

    Example response (201 Created)::

        {
          "message": "Item successfully created and saved.",
          "item_id": 1,
          "name": "Widget",
          "price": 9.99,
          "in_stock": true
        }
    """
    try:
        new_item = create_item_in_db(item)
        return {
            "message": "Item successfully created and saved.",
            "item_id": new_item.id,
            "name": new_item.name,
            "price": new_item.price,
            "in_stock": new_item.in_stock
        }
    except HTTPException as http_err:
        # If the database function raises an HTTPException (e.g., DB error)
        raise http_err
    except Exception as e:
        # Catch any unforeseen errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request."
        )


# ---------------------------------------------------------------------------
# Run the Application
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

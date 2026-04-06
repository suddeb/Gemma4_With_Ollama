from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# --- 1. Pydantic Model (Input Validation) ---
class ItemCreate(BaseModel):
    """Defines the structure and validation for incoming item data."""
    name: str = Field(..., min_length=3, description="Name of the item.")
    price: float = Field(..., gt=0, description="Price of the item (must be positive).")
    in_stock: bool = Field(..., description="Whether the item is currently in stock.")

# --- 2. SQLAlchemy Database Setup ---
# Define the base for declarative models
Base = declarative_base()

# Define the database table model
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    in_stock = Column(Boolean, default=True)

# Initialize the SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./items.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# --- 3. FastAPI Application Setup ---
app = FastAPI()

# --- 4. Database Insertion Function ---
def create_item_in_db(item: ItemCreate) -> Item:
    """
    Handles the database insertion logic.
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


# --- 5. FastAPI Endpoint ---
@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate):
    """
    Accepts item data, validates it, and persists it to the SQLite database.
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

# --- Run the Application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
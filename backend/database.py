import sqlite3
from pathlib import Path
import hashlib
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    db_path = Path(__file__).parent / "users.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY, password TEXT)
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

    crop_db_path = Path(__file__).parent / "crop_prices.db"
    conn = sqlite3.connect(str(crop_db_path))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prices
        (crop TEXT, price REAL, date TEXT)
    ''')
    conn.commit()
    conn.close()

def clean_database():
    try:
        db_path = Path(__file__).parent / "users.db"
        if db_path.exists():
            db_path.unlink()  # Delete the database file
        init_db()  # Recreate fresh database
        logger.info("Database cleaned and reinitialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error cleaning database: {str(e)}")
        return False

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def username_exists(username: str) -> bool:
    db_path = Path(__file__).parent / "users.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username=?", (username,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def add_user(username: str, password: str) -> bool | str:
    try:
        if not username or not password:
            logger.warning(f"Invalid input attempt - username: {username}")
            return "Invalid input"

        db_path = Path(__file__).parent / "users.db"
        if not db_path.exists():
            init_db()
            logger.info("Database recreated as it was missing")

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        
        # Check username with case insensitive comparison
        c.execute("SELECT 1 FROM users WHERE LOWER(username)=LOWER(?)", (username,))
        if c.fetchone():
            conn.close()
            logger.warning(f"Username already exists attempt: {username}")
            return "Username already exists"

        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                 (username, hashed_password))
        conn.commit()
        conn.close()
        logger.info(f"User created successfully: {username}")
        return True
    except Exception as e:
        logger.error(f"Error adding user {username}: {str(e)}")
        return f"Database error: {str(e)}"

def verify_user(username: str, password: str) -> bool:
    db_path = Path(__file__).parent / "users.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
             (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_random_prices():
    crops = ['Rice', 'Wheat', 'Corn', 'Soybeans', 'Cotton']
    prices = []
    for crop in crops:
        price = round(random.uniform(1000, 5000), 2)
        prices.append({'crop': crop, 'price': price})
    return prices

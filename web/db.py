"""SQLite database layer for user authentication and model persistence."""

import json
from pathlib import Path
from typing import Optional

import aiosqlite

DB_PATH = Path(__file__).parent / "tokusan.db"


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS saved_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                training_data_path TEXT,
                class_names TEXT NOT NULL,
                classifier_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name)
            )
        """)
        await db.commit()


async def get_db() -> aiosqlite.Connection:
    """Get a database connection."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def create_user(username: str, password_hash: str) -> int:
    """Create a new user and return the user ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        await db.commit()
        return cursor.lastrowid


async def get_user_by_username(username: str) -> Optional[dict]:
    """Get a user by username. Returns None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {"id": row["id"], "username": row["username"], "password_hash": row["password_hash"]}


async def save_user_model(
    user_id: int,
    name: str,
    model_path: str,
    training_data_path: Optional[str],
    class_names: list,
    classifier_type: str,
) -> int:
    """Save or update a user's model record. Returns model ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """INSERT INTO saved_models (user_id, name, model_path, training_data_path, class_names, classifier_type)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, name) DO UPDATE SET
                   model_path = excluded.model_path,
                   training_data_path = excluded.training_data_path,
                   class_names = excluded.class_names,
                   classifier_type = excluded.classifier_type,
                   created_at = CURRENT_TIMESTAMP""",
            (user_id, name, model_path, training_data_path, json.dumps(class_names), classifier_type),
        )
        await db.commit()
        return cursor.lastrowid


async def list_user_models(user_id: int) -> list:
    """List all saved models for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, classifier_type, class_names, created_at FROM saved_models WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "classifier_type": row["classifier_type"],
                "class_names": json.loads(row["class_names"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


async def get_user_model(user_id: int, model_id: int) -> Optional[dict]:
    """Get a specific saved model for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, model_path, training_data_path, class_names, classifier_type FROM saved_models WHERE id = ? AND user_id = ?",
            (model_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "model_path": row["model_path"],
            "training_data_path": row["training_data_path"],
            "class_names": json.loads(row["class_names"]),
            "classifier_type": row["classifier_type"],
        }


async def get_latest_user_model(user_id: int) -> Optional[dict]:
    """Get the most recently saved model for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, model_path, training_data_path, class_names, classifier_type FROM saved_models WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "model_path": row["model_path"],
            "training_data_path": row["training_data_path"],
            "class_names": json.loads(row["class_names"]),
            "classifier_type": row["classifier_type"],
        }


async def delete_user_model(user_id: int, model_id: int) -> bool:
    """Delete a saved model. Returns True if a row was deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM saved_models WHERE id = ? AND user_id = ?",
            (model_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0

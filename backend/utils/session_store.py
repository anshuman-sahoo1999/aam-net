import time
import logging

logger = logging.getLogger("session_store")

class InMemoryStore:
    # Structure: { session_id: { "timestamp": float, "original_img": ndarray, "motifs": list, ... } }
    _store = {}
    
    # Session timeout: 30 minutes
    SESSION_TIMEOUT = 1800

    @classmethod
    def create_session(cls, session_id: str, data: dict):
        cls.cleanup_expired()
        data["timestamp"] = time.time()
        cls._store[session_id] = data
        logger.info(f"Session {session_id} created in memory.")

    @classmethod
    def get_session(cls, session_id: str) -> dict:
        cls.cleanup_expired()
        return cls._store.get(session_id)

    @classmethod
    def update_session(cls, session_id: str, updates: dict):
        if session_id in cls._store:
            cls._store[session_id].update(updates)
            cls._store[session_id]["timestamp"] = time.time()
            logger.info(f"Session {session_id} updated.")

    @classmethod
    def delete_session(cls, session_id: str):
        if session_id in cls._store:
            del cls._store[session_id]
            logger.info(f"Session {session_id} deleted.")

    @classmethod
    def cleanup_expired(cls):
        now = time.time()
        expired_keys = [
            k for k, v in cls._store.items() 
            if now - v.get("timestamp", 0) > cls.SESSION_TIMEOUT
        ]
        for k in expired_keys:
            del cls._store[k]
            logger.info(f"Session {k} cleaned up due to expiration.")

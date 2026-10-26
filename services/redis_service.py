import json
import hashlib
import logging
import time
from typing import Any, Optional, Dict, List
import redis.asyncio as aioredis
from config import settings

logger = logging.getLogger(__name__)

# Global Redis client instance
redis_client: Optional[aioredis.Redis] = None

# High-performance in-memory fallback cache (LRU + TTL)
class InMemoryCache:
    def __init__(self, max_items: int = 1000):
        self._store: Dict[str, tuple[float, str]] = {}
        self._max_items = max_items

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        expires_at, val = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return val

    def set(self, key: str, val: str, ttl: int = 300):
        if len(self._store) >= self._max_items:
            # Evict expired items first
            now = time.time()
            expired = [k for k, (exp, _) in self._store.items() if exp < now]
            for k in expired:
                del self._store[k]
            # If still over limit, evict oldest 20%
            if len(self._store) >= self._max_items:
                keys_to_remove = list(self._store.keys())[:int(self._max_items * 0.2)]
                for k in keys_to_remove:
                    self._store.pop(k, None)
        self._store[key] = (time.time() + ttl, val)

    def delete(self, key: str):
        self._store.pop(key, None)

    def delete_prefix(self, prefix: str):
        matching = [k for k in self._store.keys() if k.startswith(prefix)]
        for k in matching:
            self._store.pop(k, None)

_fallback_cache = InMemoryCache()


async def init_redis() -> None:
    """Initializes the Redis connection pool."""
    global redis_client
    if not settings.REDIS_ENABLED:
        logger.info("Redis is disabled by configuration; using in-memory fallback cache.")
        return

    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=3.0,
            retry_on_timeout=True,
            max_connections=30
        )
        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis successfully at {settings.REDIS_URL}")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. Falling back to in-memory cache.")
        redis_client = None


async def close_redis() -> None:
    """Closes Redis connections on shutdown."""
    global redis_client
    if redis_client:
        try:
            await redis_client.aclose()
            logger.info("Redis connection closed cleanly.")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")
        finally:
            redis_client = None


def is_redis_available() -> bool:
    return redis_client is not None


# Generic JSON Cache Operations
async def cache_get(key: str) -> Optional[Any]:
    """Gets and deserializes JSON from Redis or fallback cache."""
    try:
        if redis_client:
            raw = await redis_client.get(key)
            if raw:
                return json.loads(raw)
            return None
    except Exception as e:
        logger.debug(f"Redis get error for {key}: {e}")

    # Fallback
    raw = _fallback_cache.get(key)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


async def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """Serializes value to JSON and stores in Redis or fallback cache with TTL."""
    try:
        serialized = json.dumps(value, default=str)
    except Exception as e:
        logger.error(f"Serialization error for key {key}: {e}")
        return False

    stored = False
    try:
        if redis_client:
            await redis_client.set(key, serialized, ex=ttl)
            stored = True
    except Exception as e:
        logger.debug(f"Redis set error for {key}: {e}")

    # Always keep in-memory fallback updated
    _fallback_cache.set(key, serialized, ttl=ttl)
    return stored or True


async def cache_delete(key: str) -> bool:
    """Deletes key from Redis and fallback cache."""
    try:
        if redis_client:
            await redis_client.delete(key)
    except Exception as e:
        logger.debug(f"Redis delete error for {key}: {e}")

    _fallback_cache.delete(key)
    return True


async def cache_delete_pattern(pattern: str) -> int:
    """Deletes all keys matching a pattern safely using SCAN."""
    count = 0
    prefix = pattern.rstrip("*")
    _fallback_cache.delete_prefix(prefix)

    try:
        if redis_client:
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await redis_client.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
    except Exception as e:
        logger.debug(f"Redis delete pattern error for {pattern}: {e}")

    return count


# ── Specialized Domain Cache Helpers ─────────────────────────────────────

# 1. User Session Cache
def _user_key(user_id: str) -> str:
    return f"user:session:{user_id}"

async def get_cached_user(user_id: str) -> Optional[dict]:
    return await cache_get(_user_key(user_id))

async def set_cached_user(user_id: str, user_dict: dict, ttl: int = 1800) -> bool:
    return await cache_set(_user_key(user_id), user_dict, ttl=ttl)

async def invalidate_user_cache(user_id: str) -> bool:
    return await cache_delete(_user_key(user_id))


# 2. Document & Topic Cache
def _doc_topics_key(doc_id: str) -> str:
    return f"doc:{doc_id}:topics"

def _topic_flashcards_key(topic_id: str) -> str:
    return f"topic:{topic_id}:flashcards"

def _user_docs_key(user_id: str) -> str:
    return f"docs:user:{user_id}"

async def get_cached_doc_topics(doc_id: str) -> Optional[List[dict]]:
    return await cache_get(_doc_topics_key(doc_id))

async def set_cached_doc_topics(doc_id: str, topics: List[dict], ttl: int = 3600) -> bool:
    return await cache_set(_doc_topics_key(doc_id), topics, ttl=ttl)

async def get_cached_topic_flashcards(topic_id: str) -> Optional[dict]:
    return await cache_get(_topic_flashcards_key(topic_id))

async def set_cached_topic_flashcards(topic_id: str, data: dict, ttl: int = 3600) -> bool:
    return await cache_set(_topic_flashcards_key(topic_id), data, ttl=ttl)

async def get_cached_user_documents(user_id: str) -> Optional[List[dict]]:
    return await cache_get(_user_docs_key(user_id))

async def set_cached_user_documents(user_id: str, docs: List[dict], ttl: int = 300) -> bool:
    return await cache_set(_user_docs_key(user_id), docs, ttl=ttl)

async def invalidate_user_documents(user_id: str):
    await cache_delete(_user_docs_key(user_id))


# 3. Notes Cache
def _notes_prefix(user_id: str) -> str:
    return f"notes:user:{user_id}"

async def get_cached_notes(user_id: str, query_hash: str) -> Optional[List[dict]]:
    return await cache_get(f"{_notes_prefix(user_id)}:{query_hash}")

async def set_cached_notes(user_id: str, query_hash: str, notes: List[dict], ttl: int = 180) -> bool:
    return await cache_set(f"{_notes_prefix(user_id)}:{query_hash}", notes, ttl=ttl)

async def invalidate_user_notes(user_id: str):
    await cache_delete_pattern(f"{_notes_prefix(user_id)}:*")


# 4. Quizzes Cache
def _quizzes_key(user_id: str) -> str:
    return f"quizzes:user:{user_id}"

async def get_cached_user_quizzes(user_id: str) -> Optional[List[dict]]:
    return await cache_get(_quizzes_key(user_id))

async def set_cached_user_quizzes(user_id: str, quizzes: List[dict], ttl: int = 180) -> bool:
    return await cache_set(_quizzes_key(user_id), quizzes, ttl=ttl)

async def invalidate_user_quizzes(user_id: str):
    await cache_delete(_quizzes_key(user_id))


# 5. Content-Addressed LLM Generation Cache
def make_llm_cache_key(prompt_type: str, content: str, **params) -> str:
    """Creates a deterministic SHA256 key from prompt type, text content, and params."""
    param_str = json.dumps(params, sort_keys=True)
    hasher = hashlib.sha256()
    hasher.update(prompt_type.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(content.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(param_str.encode("utf-8"))
    return f"llm:{prompt_type}:{hasher.hexdigest()}"

async def get_cached_llm_result(prompt_type: str, content: str, **params) -> Optional[Any]:
    key = make_llm_cache_key(prompt_type, content, **params)
    return await cache_get(key)

async def set_cached_llm_result(prompt_type: str, content: str, result: Any, ttl: int = 86400, **params) -> bool:
    key = make_llm_cache_key(prompt_type, content, **params)
    return await cache_set(key, result, ttl=ttl)

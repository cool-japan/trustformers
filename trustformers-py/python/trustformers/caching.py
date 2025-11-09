"""
Advanced caching system for TrustformeRS

Provides efficient caching for models, tokenizers, and results to improve performance.
"""

import os
import json
import pickle
import hashlib
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Callable, Tuple, List
from dataclasses import dataclass
from functools import wraps
from weakref import WeakValueDictionary
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import sqlite3

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access count and timestamp."""
        self.access_count += 1
        self.timestamp = time.time()


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 128, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            entry.touch()
            
            self._stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size estimate
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.ttl
            )
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)
            self._stats['size'] += size_bytes
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats['size'] -= entry.size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)
            self._stats['evictions'] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats['size'] = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self._stats['hits'] / 
                       (self._stats['hits'] + self._stats['misses']) 
                       if self._stats['hits'] + self._stats['misses'] > 0 else 0)
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'size': len(self._cache),  # Number of entries, not bytes
                'entries': len(self._cache),
                'hit_rate': hit_rate,
                'size_bytes': self._stats['size'],  # Actual size in bytes
                'size_mb': self._stats['size'] / (1024 * 1024)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (alias for stats)."""
        return self.stats()
    
    @property
    def current_size(self) -> int:
        """Get current number of entries in cache."""
        with self._lock:
            return len(self._cache)
    
    def __len__(self) -> int:
        """Get current number of entries in cache."""
        with self._lock:
            return len(self._cache)


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 1024, ttl: Optional[float] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "trustformers_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl = ttl
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()
        self._lock = threading.RLock()
    
    def _init_db(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                filename TEXT,
                timestamp REAL,
                size_bytes INTEGER,
                access_count INTEGER DEFAULT 0,
                ttl REAL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT filename, timestamp, ttl, access_count FROM cache_entries WHERE key = ?',
                (key,)
            )
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return None
            
            filename, timestamp, ttl, access_count = result
            
            # Check if expired
            if ttl and time.time() - timestamp > ttl:
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
                conn.close()
                
                # Remove file
                filepath = self.cache_dir / filename
                if filepath.exists():
                    filepath.unlink()
                
                return None
            
            # Load from file
            filepath = self.cache_dir / filename
            if not filepath.exists():
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
                conn.close()
                return None
            
            try:
                with open(filepath, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access count
                cursor.execute(
                    'UPDATE cache_entries SET access_count = access_count + 1 WHERE key = ?',
                    (key,)
                )
                conn.commit()
                conn.close()
                
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
                conn.close()
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Put value in disk cache."""
        with self._lock:
            # Generate filename
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            filename = f"{key_hash}.pkl"
            filepath = self.cache_dir / filename
            
            # Serialize and save
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = filepath.stat().st_size
                
                # Update database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Use instance TTL if not specified
                effective_ttl = ttl if ttl is not None else self.ttl
                metadata_json = json.dumps(metadata) if metadata else None
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, filename, timestamp, size_bytes, access_count, ttl, metadata)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                ''', (key, filename, time.time(), size_bytes, effective_ttl, metadata_json))
                
                conn.commit()
                conn.close()
                
                # Clean up if needed
                self._cleanup_if_needed()
                
            except Exception as e:
                logger.warning(f"Failed to save cache entry {key}: {e}")
                if filepath.exists():
                    filepath.unlink()
    
    def _cleanup_if_needed(self):
        """Clean up old entries if cache is too large."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT SUM(size_bytes) FROM cache_entries')
        total_size = cursor.fetchone()[0] or 0
        
        if total_size > self.max_size_mb * 1024 * 1024:
            # Remove least recently used entries
            cursor.execute('''
                SELECT key, filename FROM cache_entries 
                ORDER BY access_count ASC, timestamp ASC
                LIMIT ?
            ''', (max(1, len(self._get_all_keys()) // 4),))
            
            to_remove = cursor.fetchall()
            for key, filename in to_remove:
                filepath = self.cache_dir / filename
                if filepath.exists():
                    filepath.unlink()
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
        
        conn.commit()
        conn.close()
    
    def _get_all_keys(self) -> List[str]:
        """Get all cache keys."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT key FROM cache_entries')
        keys = [row[0] for row in cursor.fetchall()]
        conn.close()
        return keys
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT filename FROM cache_entries')
            filenames = [row[0] for row in cursor.fetchall()]
            
            # Remove files
            for filename in filenames:
                filepath = self.cache_dir / filename
                if filepath.exists():
                    filepath.unlink()
            
            cursor.execute('DELETE FROM cache_entries')
            conn.commit()
            conn.close()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*), SUM(size_bytes), AVG(access_count) FROM cache_entries')
            count, total_size, avg_access = cursor.fetchone()
            
            conn.close()
            
            return {
                'entries': count or 0,
                'size_mb': (total_size or 0) / (1024 * 1024),
                'avg_access_count': avg_access or 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (alias for stats)."""
        return self.stats()
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cached entry."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT metadata, timestamp FROM cache_entries WHERE key = ?',
                (key,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            metadata_json, timestamp = result
            if metadata_json:
                metadata = json.loads(metadata_json)
                metadata['timestamp'] = timestamp
                return metadata
            else:
                return {'timestamp': timestamp}
    
    def cleanup(self) -> None:
        """Clean up expired entries."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Find expired entries
            cursor.execute('''
                SELECT key, filename FROM cache_entries 
                WHERE ttl IS NOT NULL AND (timestamp + ttl) < ?
            ''', (time.time(),))
            
            expired_entries = cursor.fetchall()
            
            # Remove expired entries
            for key, filename in expired_entries:
                filepath = self.cache_dir / filename
                if filepath.exists():
                    filepath.unlink()
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            
            conn.commit()
            conn.close()


class ModelCache:
    """Specialized cache for machine learning models."""
    
    def __init__(self, 
                 memory_cache_size: int = 10,
                 disk_cache_size_mb: int = 2048,
                 cache_dir: Optional[str] = None,
                 max_memory_mb: Optional[int] = None,
                 max_disk_mb: Optional[int] = None):
        # Support both old and new parameter names for backward compatibility
        if max_memory_mb is not None:
            memory_cache_size = max_memory_mb
        if max_disk_mb is not None:
            disk_cache_size_mb = max_disk_mb
            
        # Store parameters as properties
        self.max_memory_mb = memory_cache_size
        self.max_disk_mb = disk_cache_size_mb
            
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        self.disk_cache = DiskCache(cache_dir=cache_dir, max_size_mb=disk_cache_size_mb)
        self.weak_refs = WeakValueDictionary()
        self._lock = threading.RLock()
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            # Check weak references first (fastest)
            if model_key in self.weak_refs:
                return self.weak_refs[model_key]
            
            # Check memory cache
            model = self.memory_cache.get(model_key)
            if model is not None:
                self.weak_refs[model_key] = model
                return model
            
            # Check disk cache
            model = self.disk_cache.get(model_key)
            if model is not None:
                self.memory_cache.put(model_key, model)
                self.weak_refs[model_key] = model
                return model
            
            return None
    
    def put_model(self, model_key: str, model: Any, persist_to_disk: bool = True) -> None:
        """Put model in cache."""
        with self._lock:
            self.memory_cache.put(model_key, model)
            self.weak_refs[model_key] = model
            
            if persist_to_disk:
                # Use ThreadPoolExecutor for async disk save
                executor = ThreadPoolExecutor(max_workers=1)
                executor.submit(self.disk_cache.put, model_key, model)
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self.memory_cache.clear()
            self.disk_cache.clear()
            self.weak_refs.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'memory_cache': self.memory_cache.stats(),
                'disk_cache': self.disk_cache.stats(),
                'weak_refs': len(self.weak_refs)
            }
    
    def put(self, key: str, model: Any) -> None:
        """Put model in cache (alias for put_model)."""
        self.put_model(key, model)
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache (alias for get_model)."""
        return self.get_model(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (alias for stats)."""
        return self.stats()


class TokenizerCache:
    """Specialized cache for tokenizers."""
    
    def __init__(self, max_size: int = 50, ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = LRUCache(max_size=max_size, ttl=ttl)
        self._lock = threading.RLock()
    
    def get_tokenizer(self, tokenizer_key: str) -> Optional[Any]:
        """Get tokenizer from cache."""
        with self._lock:
            return self.cache.get(tokenizer_key)
    
    def put_tokenizer(self, tokenizer_key: str, tokenizer: Any) -> None:
        """Put tokenizer in cache."""
        with self._lock:
            self.cache.put(tokenizer_key, tokenizer)
    
    def clear(self) -> None:
        """Clear tokenizer cache."""
        with self._lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get tokenizer cache statistics."""
        with self._lock:
            return self.cache.stats()
    
    def put(self, key: str, tokenizer: Any, config: Optional[Dict[str, Any]] = None) -> None:
        """Put tokenizer in cache (alias for put_tokenizer)."""
        self.put_tokenizer(key, tokenizer)
        # Store config separately if provided
        if config:
            if not hasattr(self, '_config_store'):
                self._config_store = {}
            self._config_store[key] = config
    
    def get(self, key: str) -> Optional[Any]:
        """Get tokenizer from cache (alias for get_tokenizer)."""
        return self.get_tokenizer(key)
    
    def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a cached tokenizer."""
        if hasattr(self, '_config_store'):
            return self._config_store.get(key)
        return None


class ResultCache:
    """Cache for function results with memoization."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = LRUCache(max_size=max_size, ttl=ttl)
        self._lock = threading.RLock()
    
    def memoize(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for memoizing function results."""
        import inspect
        
        # Get function signature for argument normalization
        sig = inspect.signature(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Normalize arguments to canonical form
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Create cache key from normalized arguments
            key = self._create_key(func.__name__, bound_args.args, bound_args.kwargs)
            
            # Try to get from cache
            result = self.cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            self.cache.put(key, result)
            return result
        
        return wrapper
    
    def _create_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Create cache key from function name and arguments."""
        try:
            # Normalize kwargs to handle different orderings
            normalized_kwargs = tuple(sorted(kwargs.items()))
            key_data = (func_name, args, normalized_kwargs)
            key_str = str(key_data)
            return hashlib.sha256(key_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return f"{func_name}_{hash((args, tuple(sorted(kwargs.items()))))}"
    
    def clear(self) -> None:
        """Clear result cache."""
        with self._lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get result cache statistics."""
        with self._lock:
            return self.cache.stats()
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            self.cache.put(key, value)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            return self.cache.get(key)
    
    def generate_key(self, func_name: str, args: Tuple = (), kwargs: Optional[Dict] = None) -> str:
        """Generate cache key from function name and arguments."""
        return self._create_key(func_name, args, kwargs or {})


# Global cache instances
_model_cache = ModelCache()
_tokenizer_cache = TokenizerCache()
_result_cache = ResultCache()


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    return _model_cache


def get_tokenizer_cache() -> TokenizerCache:
    """Get global tokenizer cache instance."""
    return _tokenizer_cache


def get_result_cache() -> ResultCache:
    """Get global result cache instance."""
    return _result_cache


def memoize(func: Optional[Callable[..., T]] = None, *, max_size: int = 1000, ttl: Optional[float] = None) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """Decorator for memoizing function results."""
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        # Create a dedicated cache for this function if custom parameters are provided
        if max_size != 1000 or ttl is not None:
            cache = ResultCache(max_size=max_size, ttl=ttl)
            return cache.memoize(f)
        else:
            return _result_cache.memoize(f)
    
    if func is None:
        # Called with parameters: @memoize(max_size=100)
        return decorator
    else:
        # Called without parameters: @memoize
        return decorator(func)


def cache_model(model_key: str, model: Any, persist_to_disk: bool = True) -> bool:
    """Cache a model."""
    try:
        _model_cache.put_model(model_key, model, persist_to_disk)
        return True
    except Exception as e:
        logger.warning(f"Failed to cache model {model_key}: {e}")
        return False


def get_cached_model(model_key: str) -> Optional[Any]:
    """Get a cached model."""
    return _model_cache.get_model(model_key)


def cache_tokenizer(tokenizer_key: str, tokenizer: Any) -> bool:
    """Cache a tokenizer."""
    try:
        _tokenizer_cache.put_tokenizer(tokenizer_key, tokenizer)
        return True
    except Exception as e:
        logger.warning(f"Failed to cache tokenizer {tokenizer_key}: {e}")
        return False


def get_cached_tokenizer(tokenizer_key: str) -> Optional[Any]:
    """Get a cached tokenizer."""
    return _tokenizer_cache.get_tokenizer(tokenizer_key)


def clear_all_caches() -> None:
    """Clear all caches."""
    _model_cache.clear()
    _tokenizer_cache.clear()
    _result_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return {
        'model_cache': _model_cache.stats(),
        'tokenizer_cache': _tokenizer_cache.stats(),
        'result_cache': _result_cache.stats()
    }
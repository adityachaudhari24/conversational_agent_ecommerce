"""Result Cache for Retrieval Pipeline.

This module implements an LRU cache for retrieval results to improve performance
by avoiding repeated expensive operations for the same queries.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import hashlib
import json
import logging

if TYPE_CHECKING:
    from ..pipeline import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for result cache."""
    
    enabled: bool = True
    ttl_seconds: int = 300  # 5 minutes default
    max_size: int = 1000


@dataclass
class CacheEntry:
    """Cache entry containing result and metadata."""
    
    result: 'RetrievalResult'
    created_at: datetime
    hits: int = 0


class ResultCache:
    """LRU cache for retrieval results.
    
    Provides caching functionality for retrieval results to improve performance
    by avoiding repeated expensive operations for identical queries and filters.
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize the result cache.
        
        Args:
            config: Cache configuration settings
        """
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: Dict[str, datetime] = {}  # Track access order for LRU
        
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional['RetrievalResult']:
        """Get cached result if exists and not expired.
        
        Args:
            query: The search query
            filters: Optional metadata filters applied to the search
            
        Returns:
            Cached RetrievalResult if found and not expired, None otherwise
        """
        if not self.config.enabled:
            return None
            
        cache_key = self._generate_key(query, filters)
        
        if cache_key not in self._cache:
            logger.debug(f"Cache miss for query hash: {cache_key[:8]}...")
            return None
            
        entry = self._cache[cache_key]
        
        if self._is_expired(entry):
            logger.debug(f"Cache entry expired for query hash: {cache_key[:8]}...")
            del self._cache[cache_key]
            if cache_key in self._access_order:
                del self._access_order[cache_key]
            return None
            
        # Update access order and hit count
        entry.hits += 1
        self._access_order[cache_key] = datetime.utcnow()
        
        logger.debug(f"Cache hit for query hash: {cache_key[:8]}... (hits: {entry.hits})")
        return entry.result
        
    def set(self, query: str, result: 'RetrievalResult', filters: Optional[Dict] = None) -> None:
        """Cache a retrieval result.
        
        Args:
            query: The search query
            result: The retrieval result to cache
            filters: Optional metadata filters applied to the search
        """
        if not self.config.enabled:
            return
            
        cache_key = self._generate_key(query, filters)
        
        # Check if cache is full and evict oldest entries
        if len(self._cache) >= self.config.max_size and cache_key not in self._cache:
            self._evict_oldest()
            
        # Create cache entry
        entry = CacheEntry(
            result=result,
            created_at=datetime.utcnow(),
            hits=0
        )
        
        self._cache[cache_key] = entry
        self._access_order[cache_key] = datetime.utcnow()
        
        logger.debug(f"Cached result for query hash: {cache_key[:8]}... (cache size: {len(self._cache)})")
        
    def _generate_key(self, query: str, filters: Optional[Dict]) -> str:
        """Generate cache key from query and filters.
        
        Args:
            query: The search query
            filters: Optional metadata filters
            
        Returns:
            SHA-256 hash of the query and filters combination
        """
        # Create a deterministic representation of the query and filters
        key_data = {
            'query': query.strip().lower(),  # Normalize query for consistent caching
            'filters': filters or {}
        }
        
        # Sort filters to ensure consistent key generation
        if filters:
            key_data['filters'] = dict(sorted(filters.items()))
            
        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired.
        
        Args:
            entry: The cache entry to check
            
        Returns:
            True if the entry has expired, False otherwise
        """
        if self.config.ttl_seconds <= 0:
            return False  # No expiration if TTL is 0 or negative
            
        expiry_time = entry.created_at + timedelta(seconds=self.config.ttl_seconds)
        return datetime.utcnow() > expiry_time
        
    def _evict_oldest(self) -> None:
        """Remove oldest entries when cache is full.
        
        Uses LRU (Least Recently Used) eviction policy.
        """
        if not self._access_order:
            return
            
        # Find the least recently accessed entry
        oldest_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
        
        # Remove from both cache and access order tracking
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        del self._access_order[oldest_key]
        
        logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")
        
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("Cache cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_hits = sum(entry.hits for entry in self._cache.values())
        
        return {
            'enabled': self.config.enabled,
            'size': len(self._cache),
            'max_size': self.config.max_size,
            'ttl_seconds': self.config.ttl_seconds,
            'total_hits': total_hits,
            'entries': [
                {
                    'key_hash': key[:8] + '...',
                    'created_at': entry.created_at.isoformat(),
                    'hits': entry.hits,
                    'expired': self._is_expired(entry)
                }
                for key, entry in self._cache.items()
            ]
        }
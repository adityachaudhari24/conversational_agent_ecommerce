"""Database connection and session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.utils.config import get_settings
from src.utils.logging import get_logger

from .models import Base

logger = get_logger(__name__)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
    
    @property
    def engine(self) -> AsyncEngine:
        """Get database engine, creating if necessary."""
        if self._engine is None:
            self._engine = create_async_engine(
                self.settings.database.url,
                echo=self.settings.database.echo,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections after 1 hour
            )
            logger.info("Database engine created")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get session factory, creating if necessary."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
            logger.info("Database session factory created")
        return self._session_factory
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async with db_manager.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database tables."""
    await db_manager.create_tables()


async def close_database() -> None:
    """Close database connections."""
    await db_manager.close()
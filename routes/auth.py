import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from config import settings
from security import hash_password, verify_password, create_access_token
from services.redis_service import get_cached_user, set_cached_user, invalidate_user_cache

# Updated imports pointing to folder locations
from models.auth_models import User
from schemas.auth_schemas import UserCreate, UserLogin, UserResponse, Token

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
security = HTTPBearer()


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: AsyncSession = Depends(get_db)
) -> User:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # Fast path: check Redis session cache (1ms instead of 250ms Neon round-trip)
    cached = await get_cached_user(user_id)
    if cached:
        c_at_str = cached.get("created_at")
        try:
            created_at = datetime.fromisoformat(c_at_str) if c_at_str else datetime.now(timezone.utc)
        except Exception:
            created_at = datetime.now(timezone.utc)

        user = User(
            id=uuid.UUID(cached["id"]) if isinstance(cached["id"], str) else cached["id"],
            email=cached["email"],
            full_name=cached["full_name"],
            reputation_credits=cached.get("reputation_credits", 0),
            role=cached.get("role", "student"),
            hashed_password=cached.get("hashed_password", ""),
            created_at=created_at
        )
        from sqlalchemy.orm.session import make_transient_to_detached
        make_transient_to_detached(user)
        return user

    # Cache miss: query PostgreSQL
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()

    if user is None:
        raise credentials_exception

    # Populate cache
    await set_cached_user(str(user.id), {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "reputation_credits": user.reputation_credits,
        "role": user.role,
        "hashed_password": user.hashed_password,
        "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else datetime.now(timezone.utc).isoformat()
    }, ttl=1800)

    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_in.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )

    new_user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=hash_password(user_in.password),
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Warm cache
    await set_cached_user(str(new_user.id), {
        "id": str(new_user.id),
        "email": new_user.email,
        "full_name": new_user.full_name,
        "reputation_credits": new_user.reputation_credits,
        "role": new_user.role,
        "hashed_password": new_user.hashed_password,
        "created_at": new_user.created_at.isoformat() if getattr(new_user, "created_at", None) else datetime.now(timezone.utc).isoformat()
    }, ttl=1800)

    return new_user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalars().first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})

    # Warm session cache immediately
    await set_cached_user(str(user.id), {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "reputation_credits": user.reputation_credits,
        "role": user.role,
        "hashed_password": user.hashed_password,
        "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else datetime.now(timezone.utc).isoformat()
    }, ttl=1800)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
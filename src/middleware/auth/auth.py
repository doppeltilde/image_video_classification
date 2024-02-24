from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from src.shared.shared import api_keys, use_api_keys

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)):
    if not use_api_keys:
        return api_key
    elif api_key in api_keys:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key!",
        )

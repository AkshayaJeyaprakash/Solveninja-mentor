import os
import hashlib
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    token = f"{username}:{password}"
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    expected_hash = os.getenv("ADMIN_CREDENTIALS")
    if token_hash != expected_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

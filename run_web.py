#!/usr/bin/env python
"""Run the Tokusan web application."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable for development
    )

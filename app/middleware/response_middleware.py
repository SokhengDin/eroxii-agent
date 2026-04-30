from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Any
import json
import time

from app.schema.base_schema import RESPONSE_SCHEMA
from app.core.logger import logger

class ResponseMiddleware(BaseHTTPMiddleware):
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs", "/redoc", "/openapi.json", "/favicon.ico",
            "/health", "/metrics", "/static"
            , "/api/v1/file/download"
        ]
    
    async def dispatch(self, request: Request, call_next):
        if self._should_exclude_path(request.url.path):
            return await call_next(request)

        start   = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} ({duration:.1f}ms)"
        )

        # if isinstance(response, StreamingResponse):
        #     return response
        
        if hasattr(response, 'body_iterator'):
            return await self._format_streaming_response(response)
        elif hasattr(response, 'body'):
            return await self._format_body_response(response)
        
        return response
    
    async def _format_streaming_response(self, response: Response):
        try:
            content_type = response.headers.get('content-type', '')

            # Skip binary content
            if self._is_binary_content_type(content_type):
                return response

            body_parts  = []
            async for chunk in response.body_iterator:
                body_parts.append(chunk)

            body        = b''.join(body_parts).decode('utf-8')

            # Check if it's JSON
            if content_type.startswith('application/json'):
                try:
                    original_data = json.loads(body)
                    
                    if self._is_already_formatted(original_data):
                    
                        return JSONResponse(
                            content     = original_data,
                            status_code = response.status_code,
                            headers     = dict(response.headers)
                        )
                    
                    # Format the response
                    if response.status_code >= 200 and response.status_code < 300:
                        formatted_response = RESPONSE_SCHEMA(
                            status  = response.status_code,
                            message = "Success",
                            data    = original_data
                        )
                    else:
                        if isinstance(original_data, dict) and 'detail' in original_data:
                            message = str(original_data['detail'])
                            data    = {k: v for k, v in original_data.items() if k != 'detail'} if len(original_data) > 1 else None
                        else:
                            message = "Error occurred"
                            data    = original_data
                        
                        formatted_response = RESPONSE_SCHEMA(
                            status  = response.status_code,
                            message = message,
                            data    = data
                        )
                    
                    headers = dict(response.headers)
                    headers.pop('content-length', None)
                    
                    return JSONResponse(
                        content     = formatted_response.model_dump(mode="json"),
                        status_code = response.status_code,
                        headers     = headers
                    )
                    
                except json.JSONDecodeError:
                    pass
            
            return Response(
                content     = body,
                status_code = response.status_code,
                headers     = dict(response.headers),
                media_type  = response.headers.get('content-type')
            )
            
        except Exception as e:
            logger.error(f"Error formatting streaming response: {str(e)}")
            return response
    
    async def _format_body_response(self, response: Response):
        try:
            if hasattr(response, 'body') and response.body:
                content_type = response.headers.get('content-type', '')

                # Skip binary content
                if self._is_binary_content_type(content_type):
                    return response

                body = response.body.decode('utf-8')

                if content_type.startswith('application/json'):
                    try:
                        original_data = json.loads(body)
                        
                        if self._is_already_formatted(original_data):
                            return response
                        
                        if response.status_code >= 200 and response.status_code < 300:
                            formatted_response = RESPONSE_SCHEMA(
                                status  = response.status_code,
                                message = "Success",
                                data    = original_data
                            )
                        else:
                            if isinstance(original_data, dict) and 'detail' in original_data:
                                message = str(original_data['detail'])
                                data    = {k: v for k, v in original_data.items() if k != 'detail'} if len(original_data) > 1 else None
                            else:
                                message = "Error occurred"
                                data    = original_data
                            
                            formatted_response = RESPONSE_SCHEMA(
                                status  = response.status_code,
                                message = message,
                                data    = data
                            )
                        
                        # Remove Content-Length header to let FastAPI set it correctly
                        headers = dict(response.headers)
                        headers.pop('content-length', None)
                        
                        return JSONResponse(
                            content     = formatted_response.model_dump(mode="json"),
                            status_code = response.status_code,
                            headers     = headers
                        )
                        
                    except json.JSONDecodeError:
                        pass
                        
            return response
            
        except Exception as e:
            logger.error(f"Error formatting body response: {str(e)}")
            return response
    
    def _should_exclude_path(self, path: str) -> bool:
        return any(path.startswith(exclude_path) for exclude_path in self.exclude_paths)
    
    def _is_already_formatted(self, data: Any) -> bool:
        if isinstance(data, dict):
            return all(key in data for key in ['status', 'message']) and 'data' in data
        return False

    def _is_binary_content_type(self, content_type: str) -> bool:
        """Check if content type is binary and should not be formatted"""
        binary_types = [
            'image/'
            , 'video/'
            , 'audio/'
            , 'application/pdf'
            , 'application/zip'
            , 'application/octet-stream'
            , 'application/x-'
            , 'application/vnd.'
            , 'font/'
        ]
        return any(content_type.startswith(binary) for binary in binary_types)
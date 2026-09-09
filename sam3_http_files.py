"""Binary data plane on FastMCP's existing Starlette app; no video bytes in MCP."""
import errno
import hashlib
import logging
import os
from pathlib import Path
import shutil
import uuid
from urllib.parse import quote

import anyio
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse, JSONResponse

LOG = logging.getLogger('sam3-mcp.files')
CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024**3


class FileAPIError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def relative_file(root, value):
    """Reject absolute/traversal paths and canonicalize symlinks before confinement check."""
    if not value or '\x00' in value or '\\' in value:
        raise FileAPIError('Invalid relative path')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise FileAPIError('Path must be relative and cannot contain ..', 403)
    if any(part.startswith('.') for part in path.parts):
        raise FileAPIError('Temporary/hidden files are not accessible', 403)
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise FileAPIError('Path is outside the allowed root', 403)
    return resolved


def download_reference(root, path):
    relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return relative, '/download/' + quote(relative, safe='/')


class MultipartUpload:
    """Synchronous parser callbacks executed in a worker, with bounded headers and I/O.

    Only one file part named 'file' is accepted. No Starlette form()/UploadFile spool
    is used: bytes go directly onto the input filesystem, not HPC node-local /tmp.
    """
    def __init__(self, root, boundary, max_bytes):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.file_id = uuid.uuid4().hex
        self.directory = relative_file(self.root, self.file_id)
        self.directory.mkdir(mode=0o700)
        self.temporary = self.directory / '.upload.part'
        self.stream = None
        self.filename = None
        self.size = 0
        self.body_size = 0
        self.hash = hashlib.sha256()
        self.parts = 0
        self.ended = False
        self.part_ended = False
        self.header_bytes = 0
        self.header_name = bytearray()
        self.header_value = bytearray()
        self.headers = {}
        self.parser = MultipartParser(boundary, {
            'on_part_begin': self.part_begin,
            'on_header_field': self.header_field,
            'on_header_value': self.header_data,
            'on_header_end': self.header_end,
            'on_headers_finished': self.headers_finished,
            'on_part_data': self.part_data,
            'on_part_end': self.part_end,
            'on_end': self.end,
        })

    def part_begin(self):
        self.parts += 1
        if self.parts != 1:
            raise FileAPIError('Expected exactly one file part named file')

    def header_piece(self, target, data, start, end):
        self.header_bytes += end-start
        if self.header_bytes > 16384:
            raise FileAPIError('Multipart headers exceed 16 KiB', 413)
        target.extend(data[start:end])

    def header_field(self, data, start, end):
        self.header_piece(self.header_name, data, start, end)

    def header_data(self, data, start, end):
        self.header_piece(self.header_value, data, start, end)

    def header_end(self):
        key = bytes(self.header_name).lower()
        if key in self.headers:
            raise FileAPIError('Duplicate multipart header')
        self.headers[key] = bytes(self.header_value)
        self.header_name.clear()
        self.header_value.clear()

    def headers_finished(self):
        disposition, options = parse_options_header(self.headers.get(b'content-disposition', b''))
        if disposition != b'form-data' or options.get(b'name') != b'file' or b'filename' not in options:
            raise FileAPIError('Expected multipart file field named file')
        try:
            name = options[b'filename'].decode('utf-8')
        except UnicodeDecodeError as exc:
            raise FileAPIError('Filename must be UTF-8') from exc
        if (not name or name.startswith('.') or any(c in name for c in '/\\')
                or any(ord(c) < 32 or ord(c) == 127 for c in name)
                or len(name.encode('utf-8')) > 240):
            raise FileAPIError('Filename must be a plain basename, at most 240 UTF-8 bytes')
        self.filename = name
        relative_file(self.root, f'{self.file_id}/{name}')
        self.stream = self.temporary.open('xb')

    def part_data(self, data, start, end):
        self.size += end-start
        if self.size > self.max_bytes:
            raise FileAPIError('Uploaded file exceeds --max-upload-bytes', 413)
        if self.stream is None:
            raise FileAPIError('File content received before file headers')
        view = memoryview(data)[start:end]
        self.stream.write(view)
        self.hash.update(view)

    def part_end(self):
        self.part_ended = True

    def end(self):
        self.ended = True

    def feed(self, data):
        self.body_size += len(data)
        if self.body_size > self.max_bytes + 65536:
            raise FileAPIError('Multipart body exceeds upload limit', 413)
        self.parser.write(data)

    def finish(self):
        self.parser.finalize()
        # python-multipart finalize() alone does not detect truncated bodies.
        if not self.ended or not self.part_ended or self.stream is None:
            raise FileAPIError('Incomplete multipart upload; final boundary is missing')
        if not self.size:
            raise FileAPIError('Uploaded file is empty')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.stream = None
        destination = relative_file(self.root, f'{self.file_id}/{self.filename}')
        self.temporary.replace(destination)  # Same Lustre directory/filesystem: atomic publication.
        return dict(ok=True, file_id=self.file_id, filename=self.filename, size=self.size,
                    path=destination.relative_to(self.root).as_posix(), sha256=self.hash.hexdigest())

    def cleanup(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        shutil.rmtree(self.directory, ignore_errors=True)


def error_response(exc):
    if isinstance(exc, FileAPIError):
        status, message = exc.status, str(exc)
    elif isinstance(exc, (MultipartParseError, ClientDisconnect)):
        status, message = 400, 'Malformed or interrupted multipart upload'
    elif isinstance(exc, FileNotFoundError):
        status, message = 404, 'File not found'
    elif isinstance(exc, PermissionError):
        status, message = 403, 'Filesystem permission denied'
    elif isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        status, message = 507, 'Upload storage is full or quota exceeded'
    else:
        LOG.exception('File API failure')
        status, message = 500, 'File operation failed; check server logs'
    return JSONResponse(dict(ok=False, error=message), status_code=status)


def register_file_routes(mcp, input_root, work_root, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES):
    input_root, work_root = Path(input_root).resolve(), Path(work_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if max_upload_bytes <= 0:
        raise ValueError('--max-upload-bytes must be positive')

    @mcp.custom_route('/upload', methods=['POST'])
    async def upload(request):
        upload_state = None
        committed = False
        try:
            kind, options = parse_options_header(request.headers.get('content-type', ''))
            boundary = options.get(b'boundary')
            if kind != b'multipart/form-data':
                raise FileAPIError('Use multipart/form-data with field file', 415)
            if not boundary or len(boundary) > 200:
                raise FileAPIError('Missing or invalid multipart boundary')
            if 'content-length' in request.headers:
                try:
                    length = int(request.headers['content-length'])
                except ValueError as exc:
                    raise FileAPIError('Invalid Content-Length') from exc
                if length < 0:
                    raise FileAPIError('Invalid Content-Length')
                if length > max_upload_bytes + 65536:
                    raise FileAPIError('Multipart body exceeds upload limit', 413)
            upload_state = await run_in_threadpool(MultipartUpload, input_root, boundary, max_upload_bytes)
            async for chunk in request.stream():
                # Backpressure: finish disk I/O before requesting the next ASGI chunk.
                for offset in range(0, len(chunk), CHUNK_SIZE):
                    await run_in_threadpool(upload_state.feed, chunk[offset:offset+CHUNK_SIZE])
            result = await run_in_threadpool(upload_state.finish)
            committed = True
            LOG.info('Uploaded file=%s bytes=%s', result['path'], result['size'])
            return JSONResponse(result, status_code=201)
        except (FileAPIError, MultipartParseError, ClientDisconnect, OSError) as exc:
            return error_response(exc)
        except Exception as exc:
            return error_response(exc)
        finally:
            if upload_state is not None and not committed:
                # Cleanup must survive a client disconnect or task cancellation.
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(upload_state.cleanup)

    @mcp.custom_route('/download/{path:path}', methods=['GET', 'HEAD'])
    async def download(request):
        try:
            path = await run_in_threadpool(relative_file, work_root, request.path_params['path'])
            stat = await run_in_threadpool(path.stat)
            if not path.is_file():
                raise FileAPIError('File not found', 404)
            # Starlette streams bounded chunks and implements Range/If-Range/HEAD.
            return FileResponse(path, filename=path.name, stat_result=stat,
                                content_disposition_type='attachment')
        except (FileAPIError, OSError, ValueError) as exc:
            return error_response(exc)

    @mcp.custom_route('/files/info/{path:path}', methods=['GET'])
    async def info(request):
        try:
            scope = request.query_params.get('root', 'work')
            if scope not in ('input', 'work'):
                raise FileAPIError('root must be input or work')
            root = input_root if scope == 'input' else work_root
            path = await run_in_threadpool(relative_file, root, request.path_params['path'])
            if not await run_in_threadpool(path.is_file):
                return JSONResponse(dict(exists=False, error='File not found'), status_code=404)
            stat = await run_in_threadpool(path.stat)
            return JSONResponse(dict(exists=True, size=stat.st_size, filename=path.name,
                                     path=path.relative_to(root).as_posix(), root=scope))
        except (FileAPIError, OSError, ValueError) as exc:
            return error_response(exc)

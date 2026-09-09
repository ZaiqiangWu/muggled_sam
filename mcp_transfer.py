#!/usr/bin/env python3
"""Run on Mac: explicit SFTP transfer, independent of SAM3 and MCP."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
import os


def quote(path):
    if any(c in str(path) for c in '\n\r\x00"\\'):
        raise ValueError('Paths must not contain newlines, quotes, NUL or backslashes')
    return '"' + str(path) + '"'


def transfer(host, direction, local, remote):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',host):
        raise ValueError('Use a trusted SSH config alias or user@host')
    if not remote.startswith('/'):
        raise ValueError('Remote path must be an absolute Ubuntu path')
    local = Path(local).expanduser().resolve()
    if direction == 'upload':
        if not local.is_file():
            raise ValueError('Local input file does not exist')
        # remote is an existing destination directory; a unique filename avoids overwrites.
        destination = remote.rstrip('/')+'/'+uuid.uuid4().hex+local.suffix
        command = f'put {quote(local)} {quote(destination)}\n'
        subprocess.run(['sftp','-oBatchMode=yes','-b','-',host],input=command,text=True,check=True)
        return destination
    if local.exists():
        raise FileExistsError(local)
    local.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sam3-download-',dir=local.parent) as tmp:
        temporary = Path(tmp)/'download'
        command = f'get {quote(remote)} {quote(temporary)}\n'
        subprocess.run(['sftp','-oBatchMode=yes','-b','-',host],input=command,text=True,check=True)
        os.link(temporary,local)  # Atomic no-clobber publication on the same filesystem.
    return str(local)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('direction',choices=['upload','download'])
    parser.add_argument('--host',required=True,help='SSH config alias; ProxyJump is supported via ~/.ssh/config')
    parser.add_argument('--local',required=True)
    parser.add_argument('--remote',required=True,help='Existing directory for upload; archive path for download')
    args = parser.parse_args()
    print(transfer(args.host,args.direction,args.local,args.remote))

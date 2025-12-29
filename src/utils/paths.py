import os


def normalize_path(path: str, mode: str = 'auto') -> str:
    """
    Normalize dataset path between WSL-style (/mnt/c/...) and Windows (C:\\...).
    mode: 'auto' detects environment; 'windows' or 'posix' forces conversion.
    """
    if mode == 'posix':
        return path.replace('C:\\', '/mnt/c/').replace('\\', '/').lower()
    if mode == 'windows' or (mode == 'auto' and os.name == 'nt'):
        if path.startswith('/mnt/'):  # /mnt/c/... -> C:\...
            parts = path.split('/')
            drive = parts[2].upper() + ':\\'
            rest = '\\'.join(parts[3:])
            return drive + rest
        return path
    # auto on posix
    return path


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False
import psutil
import xattr


def set_is_active_flag(file_path: str, is_active: bool) -> None:
    if psutil.LINUX:
        xattr.setxattr(file_path, 'user.is_Active', str(is_active).encode())

def get_is_active_flag(file_path: str) -> bool:
    if psutil.LINUX:
        try:
            attr_value = xattr.getxattr(file_path, 'user.is_Active').decode()
            if isinstance(attr_value, str):
                return attr_value == 'True'
            else:
                return attr_value
        except OSError:
            return False

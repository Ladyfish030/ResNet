import importlib.machinery
import os

from torch.hub import _get_torch_home

# 定义_HOME变量，指向torch home目录下的datasets/vision子目录
_HOME = os.path.join(_get_torch_home(), "datasets", "vision")
# 定义_USE_SHARDED_DATASETS变量，用于控制是否使用分片数据集
_USE_SHARDED_DATASETS = False

# 定义一个函数，用于从远程位置下载文件
def _download_file_from_remote_location(fpath: str, url: str) -> None:
    pass

# 定义一个函数，用于检查远程位置是否可用
def _is_remote_location_available() -> bool:
    return False

# 尝试从torch.hub导入load_state_dict_from_url函数
try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    # 如果导入失败，从torch.utils.model_zoo导入load_url函数，并将其重命名为load_state_dict_from_url
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

# 定义一个函数，用于获取扩展的路径
def _get_extension_path(lib_name):

    # 获取当前文件的目录
    lib_dir = os.path.dirname(__file__)
    # 判断操作系统是否为Windows
    if os.name == "nt":
        # 在Windows平台上注册DLL目录
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        # 判断是否支持AddDllDirectory函数
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        # 保存当前的错误模式
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            # 如果支持AddDllDirectory函数，将其返回类型设置为void *
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        # 将当前目录添加到DLL搜索路径中
        os.add_dll_directory(lib_dir)

        # 恢复之前的错误模式
        kernel32.SetErrorMode(prev_error_mode)

    # 获取ExtensionFileLoader和EXTENSION_SUFFIXES
    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    # 创建一个FileFinder对象，用于查找扩展
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    # 使用FileFinder查找指定的扩展
    ext_specs = extfinder.find_spec(lib_name)
    # 如果没有找到扩展，抛出ImportError异常
    if ext_specs is None:
        raise ImportError

    # 返回扩展的路径
    return ext_specs.origin

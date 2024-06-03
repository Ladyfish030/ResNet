import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont


__all__ = [
    "make_grid",
    "save_image",
    "draw_bounding_boxes",
    "draw_segmentation_masks",
    "draw_keypoints",
    "flow_to_image",
]


# 定义一个函数，用于将张量转换为图像网格
@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # 如果tensor是一个列表，则将其转换为4D mini-batch张量
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    # 确保tensor是4D张量
    if tensor.dim() == 2:  # 单张图像 H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # 单张图像
        if tensor.size(0) == 1:  # 如果单通道，则转换为3通道
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # 单通道图像
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # 避免修改张量
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # 遍历mini-batch维度
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    # 确保tensor是4D张量
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # 创建网格
    nmaps = tensor.size(0)  # 获取tensor中的图像数量（即batch size）
    xmaps = min(nrow, nmaps)  # 计算网格的列数，取nrow和nmaps中较小的值
    ymaps = int(math.ceil(float(nmaps) / xmaps))  # 计算网格的行数，使用上舍入确保所有图像都能被包含
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)  # 获取单个图像的高度和宽度，加上padding
    num_channels = tensor.size(1)  # 获取图像的通道数（例如3通道的RGB图像）
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding),
                           pad_value)  # 创建一个新的tensor来作为网格，大小为通道数x网格高度x网格宽度，用pad_value填充
    k = 0  # 初始化一个计数器，用来遍历所有的图像
    for y in range(ymaps):  # 遍历网格的行
        for x in range(xmaps):  # 遍历网格的列
            if k >= nmaps:  # 如果计数器k大于图像数量，说明所有图像已经被复制到网格中，退出循环
                break
            # 将当前图像复制到网格的对应位置
            # narrow函数用于选择一个子区域，这里选择的是当前(x,y)位置对应的区域
            # copy_()方法用于将当前图像tensor[k]复制到选择的子区域
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1  # 移动到下一个图像
    return grid  # 返回填充好的网格tensor


# 定义一个函数，用于将张量保存为图像文件
@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    # 确保tensor是一个张量或张量列表
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    if not isinstance(tensor, (torch.Tensor, list)):
        raise TypeError(f"The tensor must be a tensor or a list of tensors, got {type(tensor)}")

    # 创建一个网格，如果tensor是一个列表，则将它们合并为一个网格
    grid = make_grid(tensor, **kwargs)
    # 添加0.5到每个像素值，以四舍五入到最近的整数
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # 将张量转换为PIL图像
    im = Image.fromarray(ndarr)
    # 保存图像到文件
    im.save(fp, format=format)


# 定义一个函数，用于在图像上绘制边界框
@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:

    # 导入PIL相关的函数
    import torchvision.transforms.v2.functional as F  # noqa

    # 确保image是一个tensor，并且是uint8或浮点类型，维度为3（代表RGB图像）
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    # 获取边界框的数量
    num_boxes = boxes.shape[0]

    # 如果没有提供标签，则使用None作为默认标签
    if labels is None:
        labels = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    # 解析颜色参数，确保每个边界框都有一个颜色
    colors = _parse_colors(colors, num_objects=num_boxes)

    # 如果没有提供字体，则使用默认字体
    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # 处理灰度图像
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    # 保存原始图像的数据类型
    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        image = F.to_dtype(image, dtype=torch.uint8, scale=True)

    # 将tensor图像转换为PIL图像
    img_to_draw = F.to_pil_image(image)
    # 将边界框转换为整数列表
    img_boxes = boxes.to(torch.int64).tolist()

    # 创建一个用于绘制的图像副本
    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    # 遍历每个边界框
    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

    out = F.pil_to_tensor(img_to_draw)
    if original_dtype.is_floating_point:
        out = F.to_dtype(out, dtype=original_dtype, scale=True)
    return out


# 定义一个函数，用于在图像上绘制分割掩码
@torch.no_grad()
def draw_segmentation_masks(
        image: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.8,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
) -> torch.Tensor:
    # 确保image是一个tensor，并且是uint8或浮点类型，维度为3（代表RGB图像）
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # 如果掩码masks的维度为2，则将其扩展为3维，增加一个批次维度
    if masks.ndim == 2:
        masks = masks[None, :, :]

    # 确保掩码masks的形状为(H, W)或(batch_size, H, W)
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")

    # 确保掩码masks的数据类型为布尔类型
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")

    # 确保掩码masks的高度和宽度与图像相同
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    # 获取掩码的数量
    num_masks = masks.size()[0]

    # 计算重叠掩码的区域
    overlapping_masks = masks.sum(dim=0) > 1

    # 如果没有提供掩码，则发出警告并返回原始图像
    if num_masks == 0:
        warnings.warn("masks doesn't contain any mask. No mask was drawn")
        return image

    # 保存原始图像的数据类型
    original_dtype = image.dtype

    # 解析颜色参数，确保每个掩码都有一个颜色
    colors = [
        torch.tensor(color, dtype=original_dtype, device=image.device)
        for color in _parse_colors(colors, num_objects=num_masks, dtype=original_dtype)
    ]

    # 创建一个用于绘制的图像副本
    img_to_draw = image.detach().clone()

    # 遍历每个掩码和颜色，将掩码区域绘制到图像上
    for mask, color in zip(masks, colors):
        img_to_draw[:, mask] = color[:, None]

    # 将重叠掩码的区域设置为黑色或透明
    img_to_draw[:, overlapping_masks] = 0

    # 根据透明度alpha混合原始图像和绘制图像
    out = image * (1 - alpha) + img_to_draw * alpha

    # 将输出图像的数据类型转换回原始类型，并返回
    # Note: at this point, out is a float tensor in [0, 1] or [0, 255] depending on original_dtype
    return out.to(original_dtype)

# 定义一个函数，用于在图像上绘制关键点
@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 确保image是一个tensor，并且是uint8或浮点类型，维度为3（代表RGB图像）
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # 确保关键点keypoints的形状为(num_instances, K, 2)
    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    # 如果没有提供可见性visibility，则默认所有关键点都可见
    if visibility is None:
        visibility = torch.ones(keypoints.shape[:-1], dtype=torch.bool)
    # 调整visibility的形状以匹配关键点的形状
    visibility = visibility.squeeze(-1)
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be of shape (num_instances, K). Got ndim={visibility.ndim}")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError(
            "keypoints and visibility must have the same dimensionality for num_instances and K. "
            f"Got {visibility.shape = } and {keypoints.shape = }"
        )

    # 如果图像是浮点类型，转换为uint8类型
    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        from torchvision.transforms.v2.functional import to_dtype
        image = to_dtype(image, dtype=torch.uint8, scale=True)

    # 将tensor图像转换为PIL图像
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    # 将关键点转换为整数列表
    img_kpts = keypoints.to(torch.int64).tolist()
    # 将可见性转换为布尔列表
    img_vis = visibility.cpu().bool().tolist()

    # 遍历每个实例的关键点和可见性
    for kpt_inst, vis_inst in zip(img_kpts, img_vis):
        for kpt_coord, kp_vis in zip(kpt_inst, vis_inst):
            if not kp_vis:
                continue
            # 计算关键点椭圆的边界
            x1 = kpt_coord[0] - radius
            x2 = kpt_coord[0] + radius
            y1 = kpt_coord[1] - radius
            y2 = kpt_coord[1] + radius
            # 在图像上绘制关键点椭圆
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        # 如果提供了连接关系，绘制关键点之间的连接线
        if connectivity:
            for connection in connectivity:
                if (not vis_inst[connection[0]]) or (not vis_inst[connection[1]]):
                    continue
                start_pt_x, start_pt_y = kpt_inst[connection[0]]
                end_pt_x, end_pt_y = kpt_inst[connection[1]]
                # 在图像上绘制连接线
                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    # 将PIL图像转换回tensor
    out = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
    # 如果原始图像是浮点类型，将绘制后的图像转换回原始类型
    if original_dtype.is_floating_point:
        out = to_dtype(out, dtype=original_dtype, scale=True)
    return out


# Flow visualization code adapted from https://github.com/tomrunia/OpticalFlow_Visualization
# 定义将光流转换为图像的函数
@torch.no_grad()  # 确保在计算时不跟踪梯度
def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    if flow.dtype != torch.float:  # 检查输入光流的类型是否为浮点数
        raise ValueError(f"Flow should be of dtype torch.float, got {flow.dtype}.")

    orig_shape = flow.shape  # 保存原始光流形状
    if flow.ndim == 3:  # 如果光流只有三个维度（2个方向，高度，宽度），增加一个批次维度
        flow = flow[None]

    if flow.ndim != 4 or flow.shape[1] != 2:  # 检查光流是否有正确的维度（批次，2个方向，高度，宽度）
        raise ValueError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")

    max_norm = torch.sum(flow**2, dim=1).sqrt().max()  # 计算光流的最大范数
    epsilon = torch.finfo((flow).dtype).eps  # 获取浮点数的最小正数，以避免除以零
    normalized_flow = flow / (max_norm + epsilon)  # 归一化光流
    img = _normalized_flow_to_image(normalized_flow)  # 调用辅助函数将归一化光流转换为图像

    if len(orig_shape) == 3:  # 如果原始光流没有批次维度，移除图像的批次维度
        img = img[0]
    return img

# 定义一个内部函数，用于将归一化的光流转换为彩色图像
@torch.no_grad()  # 确保在计算时不跟踪梯度
def _normalized_flow_to_image(normalized_flow: torch.Tensor) -> torch.Tensor:
    # 获取光流数据的维度信息：批次大小N，通道数（这里应该是2，代表水平和垂直方向），高度H，宽度W
    N, _, H, W = normalized_flow.shape
    # 获取光流数据所在的设备，比如CPU或GPU
    device = normalized_flow.device
    # 初始化一个输出图像，大小为N x 3 x H x W，类型为无符号8位整数，即每个像素值在0-255之间，设备与光流数据相同
    flow_image = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=device)
    # 创建一个颜色轮，并将其移动到光流数据所在的设备
    colorwheel = _make_colorwheel().to(device)  # 颜色轮的形状为[55x3]
    # 获取颜色轮的列数，即颜色种类的数量
    num_cols = colorwheel.shape[0]
    # 计算光流数据的范数（长度），沿着第二个维度（通道维度）求和并开平方根
    norm = torch.sum(normalized_flow**2, dim=1).sqrt()
    # 计算光流数据的方向角度，使用arctan2函数，结果除以π将角度范围限定在[-1, 1]之间
    a = torch.atan2(-normalized_flow[:, 1, :, :], -normalized_flow[:, 0, :, :]) / torch.pi
    # 将角度映射到颜色轮的索引，通过移位和缩放将角度范围映射到[0, num_cols-1]
    fk = (a + 1) / 2 * (num_cols - 1)
    # 向下取整得到颜色轮的索引k0，转换为长整型张量
    k0 = torch.floor(fk).to(torch.long)
    # 计算颜色轮的下一个索引k1
    k1 = k0 + 1
    # 如果k1等于颜色轮的列数，说明索引超出了范围，将其重置为0
    k1[k1 == num_cols] = 0
    # 计算插值系数f，即fk与k0之间的差值
    f = fk - k0

    # 遍历颜色轮的每个通道（RGB）
    for c in range(colorwheel.shape[1]):
        # 获取当前通道的颜色
        tmp = colorwheel[:, c]
        # 根据索引k0和k1获取颜色轮中的两种颜色
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        # 根据插值系数f对两种颜色进行插值，得到最终的颜色col
        col = (1 - f) * col0 + f * col1
        # 根据光流范数norm调整颜色的亮度
        col = 1 - norm * (1 - col)
        # 将颜色转换为0-255的整数，并保存到输出图像的对应通道中
        flow_image[:, c, :, :] = torch.floor(255 * col)
    # 返回生成的彩色图像
    return flow_image

# 定义一个函数，用于生成用于光学流可视化的颜色轮
def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """

    # 定义颜色轮中每个颜色的范围
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    # 计算颜色轮的总颜色数
    ncols = RY + YG + GC + CB + BM + MR
    # 创建一个大小为ncols x 3的彩色轮
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # 填充RY部分的颜色
    colorwheel[0:RY, 0] = 255  # 红色通道设置为255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)  # 绿色通道设置为0到RY/RY的值
    col = col + RY

    # 填充YG部分的颜色
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)  # 红色通道设置为RY/RY到255
    colorwheel[col : col + YG, 1] = 255  # 绿色通道设置为255
    col = col + YG

    # 填充GC部分的颜色
    colorwheel[col : col + GC, 1] = 255  # 绿色通道设置为255
    colorwheel[col : col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)  # 蓝色通道设置为0到GC/GC的值
    col = col + GC

    # 填充CB部分的颜色
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)  # 绿色通道设置为GC/GC到255
    colorwheel[col : col + CB, 2] = 255  # 蓝色通道设置为255
    col = col + CB

    # 填充BM部分的颜色
    colorwheel[col : col + BM, 2] = 255  # 蓝色通道设置为255
    colorwheel[col : col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)  # 红色通道设置为0到BM/BM的值
    col = col + BM

    # 填充MR部分的颜色
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)  # 蓝色通道设置为BM/BM到255
    colorwheel[col : col + MR, 0] = 255  # 红色通道设置为255
    return colorwheel

# 定义一个函数，用于生成颜色调色板
def _generate_color_palette(num_objects: int):
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    # 创建一个num_objects个颜色的列表，每个颜色都是从palette中随机取样得到的
    return [tuple((i * palette) % 255) for i in range(num_objects)]


# 定义一个函数，用于解析对象的颜色规格
def _parse_colors(
    colors: Union[None, str, Tuple[int, int, int], List[Union[str, Tuple[int, int, int]]]],
    *,
    num_objects: int,
    dtype: torch.dtype = torch.uint8,
) -> List[Tuple[int, int, int]]:

    # 如果没有指定颜色，则自动生成一个颜色调色板
    if colors is None:
        colors = _generate_color_palette(num_objects)
    # 如果指定了一个颜色列表
    elif isinstance(colors, list):
        # 如果颜色列表中的颜色数量少于要着色的对象数量，则抛出异常
        if len(colors) < num_objects:
            raise ValueError(
                f"Number of colors must be equal or larger than the number of objects, but got {len(colors)} < {num_objects}."
            )
    # 如果colors不是列表、元组、字符串或None，则抛出异常
    elif not isinstance(colors, (tuple, str)):
        raise ValueError("`colors` must be a tuple or a string, or a list thereof, but got {colors}.")
    # 如果colors是一个元组但不是RGB三元组，则抛出异常
    elif isinstance(colors, tuple) and len(colors) != 3:
        raise ValueError("If passed as tuple, colors should be an RGB triplet, but got {colors}.")
    # 如果colors指定了一个颜色用于所有对象，则复制该颜色以匹配对象数量
    else:
        colors = [colors] * num_objects

    # 将字符串颜色转换为RGB元组
    colors = [ImageColor.getrgb(color) if isinstance(color, str) else color for color in colors]
    # 如果数据类型是浮点数，则将RGB值转换为0-1的范围
    if dtype.is_floating_point:
        colors = [tuple(v / 255 for v in color) for color in colors]
    # 返回RGB颜色列表
    return colors


# 定义一个函数，用于在组织内部记录API的使用情况
def _log_api_usage_once(obj: Any) -> None:
    # 提取对象的模块名
    module = obj.__module__
    # 如果模块名不是以torchvision开头，则修改为内部模块名
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    # 提取对象的类名或方法名
    name = obj.__class__.__name__
    # 如果obj是一个函数，则提取方法名
    if isinstance(obj, FunctionType):
        name = obj.__name__
    # 使用torch的C扩展库记录API使用情况
    torch._C._log_api_usage_once(f"{module}.{name}")


# 定义一个函数，用于从输入x创建一个n元组
def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    # 如果x是一个可迭代对象（例如列表、元组等），则将其转换为元组
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    # 否则，创建一个长度为n的元组，其中所有元素都是x的值
    return tuple(repeat(x, n))

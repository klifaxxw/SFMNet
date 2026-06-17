# import os
# import numpy as np
# from PIL import Image


# # ======== Potsdam 数据集路径（已为你写死）========
# ANN_ROOT = "/home/dell/x/data/potsdam/ann_dir"

# # Potsdam 六类
# CLASS_NAMES = [
#     "Impervious",   # 0
#     "Building",     # 1
#     "LowVeg",       # 2
#     "Tree",         # 3
#     "Car",          # 4
#     "Clutter"       # 5
# ]


# def get_mask_paths(ann_root, splits=('train', 'val')):
#     """从 ann_dir 下递归获取所有 PNG mask"""
#     split_dict = {}

#     for split_name in splits:
#         split_dir = os.path.join(ann_root, split_name)
#         mask_paths = []

#         if not os.path.exists(split_dir):
#             print(f"[Warning] not found: {split_dir}")
#             split_dict[split_name] = []
#             continue

#         for root, _, files in os.walk(split_dir):
#             for f in files:
#                 if f.lower().endswith(".png"):
#                     mask_paths.append(os.path.join(root, f))

#         mask_paths.sort()
#         print(f"[INFO] {split_name}: found {len(mask_paths)} masks")
#         split_dict[split_name] = mask_paths

#     return split_dict


# def compute_distribution(ann_root, num_classes=6, splits=('train', 'val'), ignore_index=255):
#     split_to_paths = get_mask_paths(ann_root, splits)
#     results = {}

#     for split, paths in split_to_paths.items():
#         class_count = np.zeros(num_classes, dtype=np.int64)
#         total_pixels = 0

#         for i, p in enumerate(paths):
#             if i % 30 == 0:
#                 print(f"[{split}] processing {i}/{len(paths)} ...")

#             mask = np.array(Image.open(p))
#             flat = mask.reshape(-1)

#             # 去除 ignore_index = 255
#             flat = flat[flat != ignore_index]

#             # 防越界
#             flat = flat[(flat >= 0) & (flat < num_classes)]

#             if flat.size == 0:
#                 continue

#             binc = np.bincount(flat, minlength=num_classes)
#             class_count += binc
#             total_pixels += flat.size

#         # 计算比例
#         ratios = class_count / total_pixels if total_pixels > 0 else np.zeros(num_classes)

#         results[split] = {
#             "counts": class_count,
#             "ratios": ratios,
#             "total": total_pixels
#         }

#     return results


# def show_results(results):
#     for split, r in results.items():
#         print("\n==============================")
#         print(f"Split: {split}")
#         print(f"Total valid pixels: {r['total']}")
#         print("==============================")

#         for cid, (cnt, ratio) in enumerate(zip(r["counts"], r["ratios"])):
#             print(f"{cid:2d} {CLASS_NAMES[cid]:12s} : {cnt:12d} px ({ratio*100:.3f}%)")


# if __name__ == "__main__":
#     results = compute_distribution(ANN_ROOT)
#     show_results(results)

import os
import numpy as np
from PIL import Image

# ======== LoveDA 数据集路径（按你的写法）========
ANN_ROOT = "/home/dell/x/data/loveda/ann_dir"

# 统计的 7 个有效类（对应原标签 1~7）
CLASS_NAMES = [
    "Background",    # orig 1 -> new 0
    "Building",      # orig 2 -> new 1
    "Road",          # orig 3 -> new 2
    "Water",         # orig 4 -> new 3
    "Barren",        # orig 5 -> new 4
    "Forest",        # orig 6 -> new 5
    "Agricultural"   # orig 7 -> new 6
]

def get_mask_paths(ann_root, splits=("train", "val")):
    split_dict = {}
    for split_name in splits:
        split_dir = os.path.join(ann_root, split_name)
        mask_paths = []
        if not os.path.exists(split_dir):
            print(f"[Warning] not found: {split_dir}")
            split_dict[split_name] = []
            continue

        for root, _, files in os.walk(split_dir):
            for f in files:
                if f.lower().endswith(".png"):
                    mask_paths.append(os.path.join(root, f))

        mask_paths.sort()
        print(f"[INFO] {split_name}: found {len(mask_paths)} masks")
        split_dict[split_name] = mask_paths
    return split_dict


def compute_distribution_loveda(
    ann_root,
    splits=("train", "val"),
    orig_ignore_values=(0, 255),   # 0 = no-data; 255 若存在也忽略
    orig_valid_min=1,
    orig_valid_max=7,             # LoveDA 有效类：1~7
):
    num_classes = orig_valid_max - orig_valid_min + 1  # 7
    split_to_paths = get_mask_paths(ann_root, splits)
    results = {}

    for split, paths in split_to_paths.items():
        class_count = np.zeros(num_classes, dtype=np.int64)
        total_pixels = 0

        for i, p in enumerate(paths):
            if i % 50 == 0:
                print(f"[{split}] processing {i}/{len(paths)} ...")

            mask = np.array(Image.open(p))
            flat = mask.reshape(-1)

            # 去除 ignore（0/255）
            for ig in orig_ignore_values:
                flat = flat[flat != ig]

            # 只保留 1~7
            flat = flat[(flat >= orig_valid_min) & (flat <= orig_valid_max)]
            if flat.size == 0:
                continue

            # 映射到 0~6 方便 bincount
            flat = flat - orig_valid_min

            binc = np.bincount(flat, minlength=num_classes)
            class_count += binc
            total_pixels += flat.size

        ratios = class_count / total_pixels if total_pixels > 0 else np.zeros(num_classes)
        results[split] = {"counts": class_count, "ratios": ratios, "total": total_pixels}

    return results


def show_results(results):
    for split, r in results.items():
        print("\n==============================")
        print(f"Split: {split}")
        print(f"Total valid pixels (1~7): {r['total']}")
        print("==============================")
        for cid, (cnt, ratio) in enumerate(zip(r["counts"], r["ratios"])):
            print(f"{cid:2d} {CLASS_NAMES[cid]:13s} : {cnt:12d} px  ({ratio*100:.3f}%)")


if __name__ == "__main__":
    results = compute_distribution_loveda(ANN_ROOT)
    show_results(results)

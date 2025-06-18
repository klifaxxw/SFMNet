from detectron2.data import MetadataCatalog, DatasetCatalog
import os

BASE_DICT = {
    "root": r"/home/dell/x/model/unet/VOCdevkit/VOC2007",
    "train": r"ImageSets/Segmentation/train.txt",
    "val": r"ImageSets/Segmentation/val.txt",
    "image_dir":"JPEGImages",
    "seg_dir": "SegmentationClass_label"
}


def load_voc_style_data(split="train"):
    """
    手动加载 VOC 风格的数据集
    """
    img_dir = os.path.join(BASE_DICT["root"], BASE_DICT["image_dir"])
    ann_dir = os.path.join(BASE_DICT["root"], BASE_DICT["seg_dir"])
    split_file = os.path.join(BASE_DICT["root"], BASE_DICT[split])

    with open(split_file) as f:
        file_ids = [x.strip() for x in f.readlines()]
    dataset_dicts = []
    for file_id in file_ids:
        record = {
            "file_name": os.path.join(img_dir, f"{file_id}.png"),
            "sem_seg_file_name": os.path.join(ann_dir, f"{file_id}.png"),
            "image_id": file_id,
        }
        dataset_dicts.append(record)
    return dataset_dicts

def register_soild_semantic():
    for s in ["train", "val"]:
        DatasetCatalog.register(
            f"voc_soild_{s}",
            lambda split=s: load_voc_style_data(split),
        )
        MetadataCatalog.get(f"voc_soild_{s}").set(
            stuff_classes=["bg", "soild"],
            stuff_colors=[[0, 0, 0],[255, 255, 0]],
            ignore_label=255,
            evaluator_type="sem_seg",
        )

if __name__ == "__main__":
    register_soild_semantic()
    print("Registered datasets:", list(DatasetCatalog.list()))
    print("Train samples:", len(DatasetCatalog.get("voc_soild_train")))
    print("Val samples:", len(DatasetCatalog.get("voc_soild_val")))
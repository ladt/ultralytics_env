from pathlib import Path
from abc import ABC, abstractmethod
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import shutil
import platform
import cv2
import pandas as pd
import numpy as np
import pickle
import json
import sys
import albumentations as A

from utils import crop_image_path, crop_image, polygon_to_bbox
from itertools import product


class CropsData(ABC):
    def __init__(self, *args, **kwargs):
        self.crop_suffix = '.png'
        pass

    @abstractmethod
    def create_data(self, out_dir: str | Path, reset: bool = False, **kwargs):
        pass

    def copy_data(self, src_dir: str | Path, dst_dir: str | Path, reset: bool = False, **kwargs):
        """
        Copy cropped files from src_dir to dst_dir, prefixing filenames with the subclass name.
        """
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        tasks = [(file.parent.name, file) for file in src_dir.rglob(f'*{self.crop_suffix}')]

        def copy_task(item):
            subdir, file = item
            dest_subdir = dst_dir / subdir
            dest_subdir.mkdir(parents=True, exist_ok=True)
            prefix = self.__class__.__name__
            dest_file = dest_subdir / f"{prefix}_{file.name}"
            if not reset and dest_file.exists():
                return
            shutil.copy2(file, dest_file)

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(copy_task, tasks), total=len(tasks), desc="Copying files", file=sys.stdout))

class SkeyeNetanyaKashiotSequencesInhouse(CropsData):
    def __init__(self, prefix_dir: str | Path, input_csv_path: str | Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_dir = prefix_dir
        self.csv_path = Path(input_csv_path)
        self.df = pd.read_csv(self.csv_path)
        # parse bbox_str into a list of coordinates
        self.df['polygon'] = self.df['bbox'].apply(
            lambda bbox_str: ast.literal_eval(
                re.sub(
                    r'\]\s*\[', '], [',  # add comma between outer elements
                    re.sub(
                        r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', r'\1, \2',  # add comma between numbers, handle negatives
                        bbox_str.replace('\n', ' ')
                    )
                )
            )
        )

    def create_data(self, out_dir: str | Path, invalid_img_arr=None, include_occluded=True, reset: bool = False, **kwargs):
        if not reset and Path(out_dir).is_dir():
            print('Data already exists')
            return

        df = self.df.copy()
        # filter out unknown classes
        df = df[df['class_final'] != 'unknown']
        # if occluded crops should be excluded, keep only rows with occluded == 'False'
        if not include_occluded:
            df = df[df['occluded'] == False]

        def process_row(row):
            image_path = row['image_path']
            # adjust image_path to use the configured prefix_dir
            _image_path = str(image_path)
            if _image_path.startswith("/media/aerospace_cvai"):
                _image_path = str(self.prefix_dir) + _image_path[len("/media/aerospace_cvai"):]
            image_path = Path(_image_path)
            # load the image and skip if it's missing or matches the invalid placeholder
            img = cv2.imread(str(image_path))
            if img is None or np.array_equal(img, invalid_img_arr):
                print(f"Warning: skipping invalid image at '{image_path}'")
                return
            polygon = [float(coord) for point in row['polygon'] for coord in point]
            bbox = polygon_to_bbox(polygon)
            crop = crop_image_path(image_path, bbox)
            if crop is not None:
                class_dir = Path(out_dir) / str(row['class_final'])
                class_dir.mkdir(parents=True, exist_ok=True)
                idx = row.name
                route_id = row['dirname']
                out_path = class_dir / f"{route_id}_{Path(image_path).stem}_idx_{idx}{self.crop_suffix}"
                cv2.imwrite(str(out_path), crop)

        # Use ThreadPoolExecutor for multithreading with progress bar
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=len(df), desc="Processing rows"))

    def copy_data(self, src_dir: str | Path, dst_dir: str | Path, reset: bool = False, **kwargs):
        return super().copy_data(src_dir, dst_dir, reset, **kwargs)


class SkeyeNetanyaKashiotSequencesDL(CropsData):
    def __init__(self, dataset_pkl_path: str | Path, input_data_base_path: str | Path, reset_pickle: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_pkl_path = Path(dataset_pkl_path)
        self.input_data_base_path = Path(input_data_base_path)
        self.reset_pickle = reset_pickle
        self.dataset_df = self.get_dataset_df()


    def get_dataset_df(self):
        if self.reset_pickle or not self.dataset_pkl_path.exists():
            self.create_dataset_pkl()
        return pd.read_pickle(self.dataset_pkl_path)
    
    def create_dataset_pkl(self):
        """
        Scan each route subdirectory under input_data_base_path for a COCO annotations_converted.json and build
        a DataFrame with columns: 'route_id', 'image', 'polygon', 'cls_str', 'occluded', 'tracked_vehicle'.
        """

        base_path = self.input_data_base_path

        def parse_route(route_dir: Path):
            ann_file = route_dir / 'annotations_converted.json'
            if not ann_file.exists():
                return pd.DataFrame(columns=['route_id','image','polygon','cls_str','occluded','tracked_vehicle'])
            coco = json.loads(ann_file.read_text())
            # map image_id -> file_name and category_id -> category name
            img_map = {img['id']: img['file_name'] for img in coco.get('images', [])}
            cat_map = {cat['id']: cat.get('name', str(cat['id'])) for cat in coco.get('categories', [])}
            rows = []
            for ann in coco.get('annotations', []):
                img_fn = img_map.get(ann['image_id'])
                if not img_fn:
                    continue
                rel_image = f"{route_dir.name}/{img_fn}"
                bbox = [float(x) for x in ann.get('bbox', [])]
                cls_str = cat_map.get(ann.get('category_id') )
                occluded = ann.get('attributes', {}).get('occluded', False)
                tracked_vehicle = bool(ann.get('attributes', {}).get('tracked_object', 0))
                track_id = ann.get('attributes', {}).get('track_id', None)
                rows.append({
                    'route_id': route_dir.name,
                    'image': rel_image,
                    'bbox': bbox,
                    'cls_str': cls_str,
                    'occluded': occluded,
                    'tracked_vehicle': tracked_vehicle,
                    'track_id': track_id
                })
            return pd.DataFrame(rows)

        # collect all route subdirectories
        routes = [d for d in base_path.iterdir() if d.is_dir()]
        # parse in parallel
        with ThreadPoolExecutor() as executor:
            dfs = list(tqdm(executor.map(parse_route, routes), total=len(routes), desc="Parsing routes", file=sys.stdout))
        full_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=['route_id','image','bbox','cls_str','occluded','tracked_vehicle','track_id']
        )
        # ensure output directory exists and save pickle
        self.dataset_pkl_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_pickle(self.dataset_pkl_path)
        return full_df

    def create_data(self, out_dir: str | Path, invalid_img_arr: np.ndarray=None, include_occluded: bool = True, reset: bool = False, **kwargs):
        """
        Crop images based on bbox from the dataset and save under out_dir/<class>/. Filters unknown and occluded if requested.
        """
        out_dir = Path(out_dir)
        if not reset and out_dir.is_dir():
            print('Data already exists')
            return

        df = self.get_dataset_df().copy()
        # filter out unknown classes
        df = df[df['cls_str'] != 'unknown']
        # filter occluded if needed
        if not include_occluded:
            df = df[df['occluded'] == False]

        def process_row(row):
            # resolve full image path
            img_path = self.input_data_base_path / row['image']
            img = cv2.imread(str(img_path))
            if img is None or np.array_equal(img, invalid_img_arr):
                print(f"Warning: skipping missing image at '{img_path}'")
                return
            bbox = row['bbox']
            crop = crop_image_path(img_path, bbox)
            if crop is not None:
                class_dir = out_dir / str(row['cls_str'])
                class_dir.mkdir(parents=True, exist_ok=True)
                idx = row.name
                route_id = row['route_id']
                out_path = class_dir / f"{route_id}_{Path(img_path).stem}_idx_{idx}{self.crop_suffix}"
                cv2.imwrite(str(out_path), crop)

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]),
                      total=len(df), desc="Processing rows", file=sys.stdout))

    def copy_data(self, src_dir: str | Path, dst_dir: str | Path, reset: bool = False, **kwargs):
        return super().copy_data(src_dir, dst_dir, reset, **kwargs)


class SkeyeNetanyaKashiotSingleFrames(CropsData):
    def __init__(self, input_data_base_path: str | Path, dataset_pkl_path: str | Path, reset_pickle: bool = False, include_occluded=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data_base_path = input_data_base_path
        self.dataset_pkl_path = dataset_pkl_path
        self.reset_pickle = reset_pickle
        self.include_occluded = include_occluded

    def get_dataset_df(self):
        if self.reset_pickle or not self.dataset_pkl_path.exists():
            df = self.create_data_df()
            df.to_pickle(self.dataset_pkl_path)
        return pd.read_pickle(self.dataset_pkl_path)

    def create_data_df(self) -> pd.DataFrame:
        # Validate merged COCO JSON and image directories
        input_path = Path(self.input_data_base_path)
        merged_json = input_path / 'merged_coco_json.json'
        assert merged_json.exists(), f"Missing merged JSON at {merged_json}"
        images_base_dirs = [input_path / d for d in ['roi_extracted_flight_sample_10_03_22_chosen_yolo',
        'roi_extracted_flight_sample_14_03_22_chosen_yolo']]
        image_dirs = [d / 'images' for d in images_base_dirs if (d / 'images').is_dir()]
        assert len(image_dirs) == 2, f"Expected 2 image dirs, found {len(image_dirs)}"

        # Load COCO annotations
        coco = json.loads(merged_json.read_text())
        # Map image IDs to filenames
        id_to_name = {img['id']: img['file_name'] for img in coco.get('images', [])}
        # Map category IDs to names
        cat_map = {cat['id']: cat.get('name', str(cat['id'])) for cat in coco.get('categories', [])}
        # Prepare split sets
        train_ids = set(coco.get('train_img_ids', []))
        val_ids = set(coco.get('test_img_ids', []))
        print(f"len train_ids: {len(train_ids)}, len val_ids: {len(val_ids)}")

        missing_count = 0
        ambiguous_count = 0
        unassigned_split_count = 0
        annotations = coco.get('annotations', [])
        def _process_ann(ann):
            nonlocal missing_count, ambiguous_count, unassigned_split_count
            empty_return_val = {'image': None, 'image_path': None, 'class': None, 'bbox': None, 'split': None, 'occluded': None}
            img_id = ann.get('image_id')
            file_name = id_to_name.get(img_id)
            assert file_name, f"No file_name for image_id {img_id}"
            # Locate the image uniquely in one of the image directories
            matches = [img_dir / file_name for img_dir in image_dirs if (img_dir / file_name).exists()]
            if len(matches) < 1:
                missing_count += 1
                return empty_return_val
            if len(matches) > 1:
                ambiguous_count += 1
                return empty_return_val
            img_path = matches[0]
            rel_path = img_path.relative_to(input_path)
            # Determine split
            if img_id in train_ids:
                split = 'train'
            elif img_id in val_ids:
                split = 'val'
            else:
                unassigned_split_count += 1
                return empty_return_val
            return {
                'image': file_name,
                'image_path': str(rel_path),
                'class': cat_map.get(ann.get('category_id'), str(ann.get('category_id'))),
                'bbox': ann.get('bbox'),
                'occluded': ann.get('attributes', {}).get('Occluded', False),
                'split': split
            }

        with ThreadPoolExecutor() as executor:
            records = [
            rec for rec in tqdm(
                executor.map(_process_ann, annotations),
                total=len(annotations),
                desc="Parsing annotations",
                file=sys.stdout
            )
            ]
        print(f"Missing annotations: {missing_count}, Ambiguous annotations: {ambiguous_count}, Unassigned annotations: {unassigned_split_count}")
        df = pd.DataFrame(records)
        # drop any rows that contain None
        df = df.dropna(how='any')
        return df

    def get_augmentations(self) -> tuple[tuple | None, tuple | None]:
        train_aug = A.Compose([
            A.Rotate(limit=(-90, 90), p=0.5, border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=A.BboxParams(
            format='coco', label_fields=['class_labels', 'object_ids'], min_visibility=0.8
        ))
        train_aug = ('Rotation', train_aug, 2)
        test_aug = None
        return train_aug, test_aug

    @staticmethod
    def save_crops(img, bboxes, transform, class_labels, object_ids, crops_stem, crops_suffix, out_dir, crops_desc=None):
        if transform is not None:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels, object_ids=object_ids, hm_metadata=None)
            img, bboxes, class_labels, object_ids = augmented['image'], augmented['bboxes'], augmented['class_labels'], augmented['object_ids']
        for bbox, cls_label, object_id in zip(bboxes, class_labels, object_ids):
            crop = crop_image(img, bbox)
            if crop is None or crop.size == 0:
                print(f"Warning: skipping missing image with object id: {object_id}")
                continue
            cls_dir = Path(out_dir / cls_label)
            cls_dir.mkdir(parents=True, exist_ok=True)
            if crops_desc is not None:
                fname = f"{crops_stem}_idx_{int(object_id)}_{crops_desc}{crops_suffix}"
            else:
                fname = f"{crops_stem}_idx_{int(object_id)}{crops_suffix}"
            cv2.imwrite(str(cls_dir / fname), crop)

    def _process_group_for_maybe_augment_and_crop(self, item, transform_name, transform, transform_count, out_dir):
        """
        Process a single image group: load image, apply augmentations, and save crops.
        """
        image_name, group = item
        assert group['image_path'].nunique() == 1, f"Multiple image_path for image {image_name}"
        rel_path = group['image_path'].iloc[0]
        img_path = Path(self.input_data_base_path) / rel_path
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: skipping missing image {img_path}")
            return
        bboxes = group['bbox'].tolist()
        class_labels = group['class'].tolist()
        object_ids = group.index.tolist()
        img_stem, img_suffix = img_path.stem, img_path.suffix
        if transform is None:
            assert transform_name == transform_count == None, "If no transform is provided, transform_name and transform_count must also be None"
            self.save_crops(img, bboxes, transform, class_labels, object_ids, img_stem, img_suffix, out_dir)        
        else:
            for i in range(transform_count):
                assert transform_name and transform_count, "If transform is provided, transform_name and transform_count must also be set"
                crops_desc = f"{transform_name}_{i}"
                self.save_crops(img, bboxes, transform, class_labels, object_ids, img_stem, img_suffix, out_dir, crops_desc)

    def maybe_augment_and_crop(self, df, aug, split_str, out_dir):
        """
        Group annotations by image, assert same image_path, aggregate classes/bboxes/idx,
        then apply augmentations (if any) and crop each bbox.
        """
        transform_name, transform, transform_count = aug if aug is not None else (None, None, None)
        tasks = list(df.groupby('image'))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(
                self._process_group_for_maybe_augment_and_crop,
                item, transform_name, transform, transform_count, out_dir
            ) for item in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{split_str} groups", file=sys.stdout):
                pass

    def create_data(self, out_dir, reset = False, **kwargs):
        if reset or not Path(out_dir).exists():
            df = self.get_dataset_df()
            if self.include_occluded == False:
                df = df[df['occluded'] == False]
            train_aug, test_aug = self.get_augmentations()
            df_train, df_val = df[df['split'] == 'train'], df[df['split'] == 'val']
            for df_split, split_aug, split_str in zip([df_train, df_val], [train_aug, test_aug], ['train', 'val']):
                base_out_dir = Path(out_dir) / split_str
                self.maybe_augment_and_crop(df_split, split_aug, split_str, base_out_dir)

    def copy_data(self, src_dir, dst_dir, reset = False, **kwargs):
        def copy_task(item):
            split, src_cls, file = item
            dst_cls = cls_map.get(src_cls, src_cls)
            dest_subdir = dst_dir / split / dst_cls
            dest_subdir.mkdir(parents=True, exist_ok=True)
            prefix = self.__class__.__name__
            dest_file = dest_subdir / f"{prefix}_{file.name}"
            if not reset and dest_file.exists():
                return
            shutil.copy2(file, dest_file)

        classes = ['bus', 'van', 'pick-up', 'private', 'truck', 'tractor', 'motorcycle']
        cls_map = {'pickup': 'pick-up', 'truck_open_body': 'truck', 'truck_closed_body': 'truck', 'concrete_mixer': 'truck'}
        crop_suffix = '.jpg'
        possible_input_dirnames = set(classes + list(cls_map.keys()))
        for split_dir in ['train', 'val']:
            src_cls_dirs = [d for d in Path(Path(src_dir) / split_dir).glob('*') if d.is_dir() and d.name in possible_input_dirnames]
            src_files = [f for d in src_cls_dirs for f in d.glob(f'*{crop_suffix}')]

            tasks = [(split_dir, file.parent.name, file) for file in src_files]

            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(copy_task, tasks), total=len(tasks), desc="Copying files", file=sys.stdout))

class SkeyeRareClassesTasks(SkeyeNetanyaKashiotSingleFrames):
    def __init__(self, input_data_base_path: str | Path, dataset_pkl_path: str | Path, flights_info_csv: str | Path, ref_img_paths: list[Path], reset_pickle: bool = False, include_occluded=False, *args, **kwargs):
        super().__init__(input_data_base_path=input_data_base_path, dataset_pkl_path=dataset_pkl_path, reset_pickle=reset_pickle, include_occluded=include_occluded, *args, **kwargs)
       
        self.flights_info_csv = flights_info_csv
        self.flight_info_df = pd.read_csv(flights_info_csv)
        self.flight_info_df['flight'] = self.flight_info_df['sfid'].apply(lambda x: x.split(':')[3].strip())
        self.ref_img_paths = ref_img_paths

        self.vmd_task_numbers = [581, 530, 529, 494, 493, 492, 491, 483]
        self.task_to_flight_and_mission = {
            465: ('704_100416', 66),
            471: ('NOA_POD2_FLIGHT1', 100),
            472: ('NOA_POD2_FLIGHT1', 100),
            481: ('NOA_POD2_FLIGHT1', 110),
            482: ('NOA_POD2_FLIGHT1', 105),
            483: ('704_100416', 66),
            484: ('NOA_Client_flight1', 70),
            489: ('Arig_Adgama', 64),
            491: ('Arig', 54),
            492: ('Arig', 55),
            493: ('Arig', 56),
            494: ('Hadran2', 79),
            500: ('Arig', 52),
            501: ('Arig', 50),
            502: ('Arig', 51),
            503: ('Arig_Adgama', 65),
            504: ('Arig_Adgama', 65),
            505: ('NOA_POD2_FLIGHT1', 101),
            506: ('NOA_POD2_FLIGHT1', 101),
            507: ('NOA_POD2_FLIGHT1', 101),
            508: ('NOA_POD2_FLIGHT1', 100),
            509: ('NOA_POD2_FLIGHT1', 100),
            510: ('NOA_POD2_FLIGHT1', 100),
            511: ('NOA_POD2_FLIGHT1', 100),
            512: ('NOA_POD2_FLIGHT1', 100),
            513: ('NOA_POD2_FLIGHT1', 105),
            514: ('NOA_POD2_FLIGHT1', 105),
            515: ('NOA_POD2_FLIGHT1', 105),
            516: ('NOA_POD2_FLIGHT1', 105),
            517: ('NOA_POD2_FLIGHT1', 106),
            518: ('NOA_POD2_FLIGHT1', 106),
            519: ('NOA_POD2_FLIGHT1', 106),
            520: ('NOA_POD2_FLIGHT1', 106),
            521: ('NOA_POD2_FLIGHT1', 112),
            526: ('Arig_Adgama', 58),
            529: ('Hadran2', 100),
            530: ('Hadran2', 113),
            531: ('NOA_Client_flight1', 72),
            577: ('FT30_V7', 35),
            581: ('Hadran2', 80),
            584: ('NOA_Client_flight1', 73),
            585: ('NOA_Client_flight1', 74),
            587: ('NOA_Client_flight1', 77),
            588: ('NOA_Client_flight1', 77),
            598: ('Hadran2', 82) # appears as 'Hadran' in CVAT task CSV, but only possible candidate in flights_info.csv is 'Hadran2"
        }
    
    def get_dataset_df(self):
        return super().get_dataset_df()
    
    @staticmethod
    def get_frame_number(image_path):
        frame_stem_elements = Path(image_path).stem.split('_')
        frame_num = frame_stem_elements[1]
        return int(frame_num)

    def get_flight_and_height_kft(self, frame_num: int, task_num: int) -> tuple[str, float]:
        flight_and_mission: tuple = self.task_to_flight_and_mission[task_num]
        flight: str = flight_and_mission[0]
        mission: int = flight_and_mission[1]
        task_df = self.flight_info_df[self.flight_info_df['flight'] == flight]
        mission_df = task_df[task_df['missionIndex'] == mission]
        matching_rows = mission_df[(mission_df['startFrame'] <= frame_num) & (mission_df['endFrame'] >= frame_num)]
        assert len(matching_rows) == 1, f"Expected 1 matching row, found {len(matching_rows)}"
        row = matching_rows.iloc[0]
        flight_code, height = row['name'], row['platformAlt']
        height_kft =  height * 3.2808399 / 10 ** 3
        return flight_code, height_kft

    def process_task_dir(self, task_dir) -> pd.DataFrame:
        task_dir = Path(task_dir)
        task_num = int(task_dir.name.split('_')[1])
        task_input_csv = task_dir / "annotation_data.csv"
        task_image_dir = task_dir / "images"
        assert task_input_csv.exists(), f"Input file {task_input_csv} does not exist"
        assert task_image_dir.exists(), f"Image directory {task_image_dir} does not exist"
        task_input_df = pd.read_csv(task_input_csv)
        image_filenames = list([path.name for path in task_image_dir.glob("*.png")])
    
        def _process_task_row(task_input_df_row):
            ret_keys = ['image', 'image_path', 'class', 'bbox', 'occluded', 'height', 'flight']
            empty_ret = {k: None for k in ret_keys}
            s3_image_path = task_input_df_row['image_path']
            image_name = Path(s3_image_path).name
            full_image_path = task_image_dir / image_name
            if image_name not in image_filenames or cv2.imread(str(full_image_path)) is None:
                return empty_ret
            class_str = task_input_df_row['category_id']
            segmentation = ast.literal_eval(task_input_df_row['segmentation'])
            attributes = ast.literal_eval(task_input_df_row['attributes'])
            occluded = attributes.get('occluded', False)

            if len(segmentation) > 0:
                segmentation = segmentation[0]
                bbox = polygon_to_bbox(segmentation)
            else:
                bbox = ast.literal_eval(task_input_df_row['bbox'])
                rotation = attributes.get('rotation', 0)
                if rotation != 0:
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    center_x = float(x1 + x2) / 2
                    center_y = float(y1 + y2) / 2
                    rect = ((center_x, center_y), (w, h), -rotation)
                    cv2_bbox = cv2.boxPoints(rect)
                    bbox = polygon_to_bbox(cv2_bbox)
            frame_num = self.get_frame_number(image_name)
            flight, height_kft = self.get_flight_and_height_kft(frame_num, task_num)
            relative_image_path = Path(task_dir.name) / "images" / image_name
            return {'image': image_name, 'image_path': str(relative_image_path), 'class': class_str, 'bbox': bbox, 'occluded': occluded, 'height': height_kft, 'flight': flight}

        # Apply row processing in parallel
        rows = [row for _, row in task_input_df.iterrows()]
        with ThreadPoolExecutor() as executor:
            processed = list(tqdm(executor.map(_process_task_row, rows), total=len(rows), desc=f"Processing rows in {task_dir.name}", file=sys.stdout))
        task_df = pd.DataFrame(processed)
        # Add task-level metadata
        task_df['task_num'] = task_num
        task_df['vmd'] = task_num in self.vmd_task_numbers
        return task_df

    def create_data_df(self) -> pd.DataFrame:
        task_dirs = [d for d in self.input_data_base_path.glob('*') if d.is_dir() and d.name.startswith('task_')]
        task_dfs = []
        for task_dir in tqdm(task_dirs, desc="Processing task directories"):
            task_df = self.process_task_dir(task_dir)
            task_dfs.append(task_df)
        all_tasks_df = pd.concat(task_dfs, ignore_index=True)
        return all_tasks_df

    def get_augmentations(self, data_split, vmd) -> tuple | None:
        reference_images = [img for img_path in self.ref_img_paths if (img := cv2.imread(str(img_path))) is not None]
        assert all(img is not None for img in reference_images), "All reference images must be valid."
        histogram_matching_transform = A.HistogramMatching(reference_images=reference_images, read_fn=lambda x: x, p=1.0, blend_ratio=(1.0, 1.0))
        rotation_transform = A.Rotate(limit=(-90, 90), p=0.5, border_mode=cv2.BORDER_CONSTANT)
        bbox_params = A.BboxParams(format='coco', label_fields=['class_labels', 'object_ids'], min_visibility=0.8)
        
        if data_split == 'train':
            if vmd:
                aug = A.Compose([
                    rotation_transform
                ], bbox_params=bbox_params)
                return ('Rotation', aug, 2)
            else:
                aug = A.Compose([
                    histogram_matching_transform,
                    rotation_transform
                ], bbox_params=bbox_params)
                return ('HistMatchRotation', aug, 2)
        if not vmd:
            aug = A.Compose([
                histogram_matching_transform
            ], bbox_params=bbox_params)
            return ('HistMatch', aug, 1)
            
        # data_split == 'val' and 'vmd' == True
        return None

    def _process_task_an(self, task, transform_name, transform, transform_count, out_dir):
        image_path, group = task
        assert group['task_num'].nunique() == 1, f"Multiple task_num for image {image_path}"
        assert group['vmd'].nunique() == 1, f"Multiple vmd for image {image_path}"
        vmd = bool(group['vmd'].iloc[0])

        full_img_path = Path(self.input_data_base_path) / image_path
        img = cv2.imread(str(full_img_path))
        if img is None:
            print(f"Warning: skipping missing image {full_img_path}")
            return
        bboxes = group['bbox'].tolist()
        class_labels = group['class'].tolist()
        object_ids = group.index.tolist()
        img_stem, img_suffix = full_img_path.stem, full_img_path.suffix
        base_crops_desc = f"vmd_{vmd}"

        if transform is None:
            assert transform_name == transform_count == None, "If no transform is provided, transform_name and transform_count must also be None"
            self.save_crops(img, bboxes, transform, class_labels, object_ids, img_stem, img_suffix, out_dir, base_crops_desc)        
        else:
            for i in range(transform_count):
                assert transform_name and transform_count, "If transform is provided, transform_name and transform_count must also be be set"
                crops_desc = f"{base_crops_desc}_{transform_name}_{i}"
                self.save_crops(img, bboxes, transform, class_labels, object_ids, img_stem, img_suffix, out_dir, crops_desc)

    def _assign_data_split(self, df: pd.DataFrame) -> pd.DataFrame:
        # assign splits per task and class
        df['data_split'] = 'train'
        seed = 42
        for (task, cls), group in df.groupby(['task_num', 'class']):
            N = len(group)
            n_val = int(N * 0.2)
            # ensure at least one validation if small group
            if 1 <= N < 5:
                n_val = 1
            # cannot exceed group size
            if n_val >= N:
                n_val = max(N - 1, 0)
            if n_val > 0:
                val_idx = group.sample(n=n_val, random_state=seed).index
                df.loc[val_idx, 'data_split'] = 'val'
        return df

    def create_data(self, out_dir, reset=False, **kwargs):
        """
        Crop and save images for rare classes, splitting per task and class into train/val, applying augmentations.
        """
        out_dir = Path(out_dir)
        if not reset and out_dir.exists():
            print('Data already exists')
            return
        df = self.get_dataset_df().copy()
        if not self.include_occluded:
            df = df[df['occluded'] == False]
        df = self._assign_data_split(df)
        df['height_category'] = df['height'].apply(lambda x: '8_16' if x < 16 else '16_25')
        # process by height category and split in one loop
        for height_cat, data_split, vmd in product(df['height_category'].unique(), df['data_split'].unique(), df['vmd'].unique()):
            df_split = df[
            (df['height_category'] == height_cat) &
            (df['data_split'] == data_split) &
            (df['vmd'] == vmd)
            ]

            split_out_dir = out_dir / height_cat / data_split
            aug = self.get_augmentations(data_split, vmd)
            transform_name, transform, transform_count = aug if aug else (None, None, None)

            # tasks = list(df_split.groupby(('image_path')))
            # for task in tasks:
            #     self._process_task_an(
            #         task, transform_name, transform, transform_count, split_out_dir
            #     )

            # group by task_num and image
            tasks = list(df_split.groupby(('image_path')))
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    self._process_task_an,
                    item, transform_name, transform, transform_count, split_out_dir
                ) for item in tasks]
                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"height={height_cat}kft, split={data_split}, vmd={vmd} groups", file=sys.stdout):
                    pass            

    def copy_data(self, src_dir, dst_dir, reset = False, **kwargs):
        def copy_task(item):
            height_cat, split, src_cls, file = item
            dst_cls = cls_map.get(src_cls, src_cls)
            dest_subdir = dst_dir / height_cat / split / dst_cls
            dest_subdir.mkdir(parents=True, exist_ok=True)
            prefix = self.__class__.__name__
            dest_file = dest_subdir / f"{prefix}_{file.name}"
            if not reset and dest_file.exists():
                return
            shutil.copy2(file, dest_file)

        classes = ['bus', 'van', 'pick-up', 'private', 'truck', 'tractor', 'motorcycle']
        cls_map = {'cement-mixer': 'truck', 'closed_truck': 'truck', 'open-truck': 'truck'}
        crop_suffix = '.png'
        possible_input_dirnames = set(classes + list(cls_map.keys()))
        for height_cat_dir in ['8_16', '16_25']:
            for split_dir in ['train', 'val']:
                src_cls_dirs = [d for d in Path(Path(src_dir) / height_cat_dir / split_dir).glob('*') if d.is_dir() and d.name in possible_input_dirnames]
                src_files = [f for d in src_cls_dirs for f in d.glob(f'*{crop_suffix}')]

                tasks = [(height_cat_dir, split_dir, file.parent.name, file) for file in src_files]

                with ThreadPoolExecutor() as executor:
                    list(tqdm(executor.map(copy_task, tasks), total=len(tasks), desc="Copying files", file=sys.stdout))

if __name__ == "__main__":
    if platform.system() == "Windows":
        prefix_dir = "//devbitshares.devbit.io/aerospace_cvai"
    elif platform.system() == "Linux":
        prefix_dir = '/media/aerospace_cvai'
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")

    invalid_img_path = Path(prefix_dir) / "shared_weights/data/raw_skeye_v3/DL_tagged/4_20390010/0040.png"
    invalid_img_arr = cv2.imread(str(invalid_img_path))

    single_dataset_base_dir = Path(prefix_dir) / 'datasets/ultralytics/skeye_classifier_multi_datasets'
    flat_dataset_base_dir = Path(prefix_dir) / 'datasets/ultralytics/skeye_classifier_flat_datasets'
    flat_dataset_8_16 = flat_dataset_base_dir / '8_16'
    flat_dataset_16_25 = flat_dataset_base_dir / '16_25'
    
    # ############### Skeye Kashiot Sequences Inhouse ##############
    df_path = Path(prefix_dir) / 'shared_weights/data/raw_skeye_v3/inhouse_data.csv'
    
    dataset_out_dir = single_dataset_base_dir / 'skeye_netanya_kashiot_sequences_inhouse'
    final_data_dir = flat_dataset_8_16 / 'train'
    
    skeye_inhouse_data = SkeyeNetanyaKashiotSequencesInhouse(prefix_dir, df_path)
    skeye_inhouse_data.create_data(out_dir=dataset_out_dir, invalid_img_arr=invalid_img_arr, reset=False, include_occluded=False)
    skeye_inhouse_data.copy_data(src_dir=dataset_out_dir, dst_dir=final_data_dir)
    
    ############### Skeye Kashiot Sequences DataLoop ##############
    input_data_base_path = Path(prefix_dir) / 'shared_weights/data/raw_skeye_v3/DL_tagged'
    dataset_pkl_path = Path(prefix_dir) / 'shared_weights/data/raw_skeye_v3' / 'DL_tagged_df.pkl'
    
    dataset_out_dir = single_dataset_base_dir / 'skeye_netanya_kashiot_sequences_DL'
    final_data_dir = flat_dataset_8_16 / 'train'
    
    skeye_DL_data = SkeyeNetanyaKashiotSequencesDL(dataset_pkl_path=dataset_pkl_path, input_data_base_path=input_data_base_path, reset_pickle=False)
    skeye_DL_data.create_data(out_dir=dataset_out_dir, invalid_img_arr=invalid_img_arr, reset=False, include_occluded=False)
    skeye_DL_data.copy_data(src_dir=dataset_out_dir, dst_dir=final_data_dir)

    ############### Skeye Kashiot Single Frames ##############
    input_data_base_path = Path(prefix_dir) / 'object_detection/skeye/skeye_KASHIYOT_single_frames'
    dataset_pkl_path = input_data_base_path / 'skeye_KASHIYOT_single_frames.pkl'
    
    dataset_out_dir = single_dataset_base_dir / 'skeye_netanya_kashiot_single_frames'
    final_data_dir = Path(prefix_dir) / flat_dataset_8_16
    
    skeye_netanya_kashiot_single_frames = SkeyeNetanyaKashiotSingleFrames(input_data_base_path=input_data_base_path, dataset_pkl_path=dataset_pkl_path, include_occluded=False, reset_pickle=False)
    skeye_netanya_kashiot_single_frames.create_data(out_dir=dataset_out_dir, reset=False)
    skeye_netanya_kashiot_single_frames.copy_data(src_dir=dataset_out_dir, dst_dir=final_data_dir)

    ############### Skeye Rare Classes ##############
    input_data_base_path = Path(prefix_dir) / 'datasets/skeye/rare_classes'
    flights_info_csv = Path(prefix_dir) / 'datasets/skeye/flights_info.csv'
    dataset_pkl_path = input_data_base_path.parent / 'rare_classes.pkl'
    ref_img_paths = list((Path(prefix_dir) / 'datasets/skeye/ref_images').rglob('*.png'))
    
    dataset_out_dir = single_dataset_base_dir / 'rare_classes'
    final_data_dir = flat_dataset_base_dir
    
    skeye_rare_classes = SkeyeRareClassesTasks(input_data_base_path=input_data_base_path, dataset_pkl_path=dataset_pkl_path, flights_info_csv=flights_info_csv, ref_img_paths=ref_img_paths, include_occluded=False, reset_pickle=False)
    skeye_rare_classes.create_data(out_dir=dataset_out_dir, reset=False)
    skeye_rare_classes.copy_data(src_dir=dataset_out_dir, dst_dir=final_data_dir)
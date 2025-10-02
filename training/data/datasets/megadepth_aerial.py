import os
import os.path as osp
import logging
import random
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from data.dataset_util import *
from data.base_dataset import BaseDataset


class MegaDepthAerialDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ROOT: str = "/home/haowei/Documents/aerial-megadepth/megadepth_aerial_processed",
        split_file: str = "train.npz",  # Your merged NPZ file
        segmentation_root: str = None,  # Optional: path to segmentation masks
        min_num_images: int = 2,
        len_train: int = 100000,
        len_test: int = 10000,
        max_depth: float = 1500.0,  # Adjusted for aerial imagery
        depth_percentile: float = 98.0,
        use_pairs: bool = True,
        expand_ratio: int = 2,
        remove_sky: bool = True,
    ):
        """Initialize the MegaDepthAerialDataset for VGGT training."""
        super().__init__(common_conf=common_conf)
        
        self.ROOT = ROOT
        self.segmentation_root = segmentation_root
        self.max_depth = max_depth
        self.depth_percentile = depth_percentile
        self.use_pairs = use_pairs
        self.expand_ratio = expand_ratio
        self.remove_sky = remove_sky
        
        # Load common configurations
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        # Set dataset length
        if split == "train":
            self.len_train = len_train
        else:
            self.len_train = len_test
            
        # Load pairs and metadata
        self._load_data(split_file)
        
        # Check which scenes actually exist on disk
        self.valid_scenes = []
        for scene in ['0000', '0001']:  # Only check for scenes you have
            scene_path = osp.join(self.ROOT, scene)
            if osp.exists(scene_path):
                self.valid_scenes.append(scene)
                logging.info(f"Found scene directory: {scene}")
        
        if not self.valid_scenes:
            raise RuntimeError(f"No valid scene directories found in {self.ROOT}")
            
        # Filter pairs to only valid scenes
        self._filter_to_valid_scenes()
        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid pairs found after filtering! Check your data.")
        
        safety_factor = 0.8  # Assume 80% of pairs will successfully load
        max_possible = int(len(self.pairs) * safety_factor)
        
        if self.training:
            if self.len_train > max_possible:
                logging.warning(
                    f"Requested len_train={self.len_train} but only ~{max_possible} pairs "
                    f"likely available. Setting len_train={max_possible}"
                )
                self.len_train = max_possible
        
        # Log final statistics
        logging.info(f"Final dataset size: {self.len_train} (from {len(self.pairs)} valid pairs)")
            
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MegaDepthAerial Data")
        logging.info(f"  - Root: {self.ROOT}")
        logging.info(f"  - Scenes: {self.valid_scenes}")
        logging.info(f"  - Pairs: {len(self.pairs)}")
        logging.info(f"  - Dataset length: {len(self)}")

    def _load_data(self, split_file):
        """Load pairs and metadata from NPZ file."""
        split_path = osp.join(self.ROOT, split_file)
        
        if not osp.exists(split_path):
            raise FileNotFoundError(f"NPZ file not found: {split_path}")
            
        with np.load(split_path, allow_pickle=True) as data:
            self.scenes = data['scenes']
            self.images = data['images'] 
            self.pairs_raw = data['pairs']  # numpy array format
            self.images_scene_name = data.get('images_scene_name', None)
            
        logging.info(f"Loaded NPZ: {len(self.scenes)} scenes, {len(self.pairs_raw)} pairs")
        
        # Convert pairs from numpy array to dict format for easier access
        self.pairs = []
        for pair in self.pairs_raw:
            # Format: [scene_id, im1_id, im2_id, score]
            self.pairs.append({
                'scene_id': int(pair[0]),
                'im1_id': int(pair[1]),
                'im2_id': int(pair[2]),
                'score': float(pair[3]) if len(pair) > 3 else 1.0
            })
        
        # Create scene-to-images mapping for random sampling
        self.scene_to_images = {}
        for img_id, img in enumerate(self.images):
            if img is not None and self.images_scene_name is not None:
                scene = str(self.images_scene_name[img_id])
                if scene not in self.scene_to_images:
                    self.scene_to_images[scene] = []
                self.scene_to_images[scene].append((img_id, str(img)))

    def _filter_to_valid_scenes(self):
        """Filter pairs to only include valid scenes that exist on disk."""
        if not hasattr(self, 'valid_scenes'):
            return
            
        # Get valid scene IDs
        valid_scene_ids = []
        for scene in self.valid_scenes:
            scene_idx = np.where(self.scenes == scene)[0]
            if len(scene_idx) > 0:
                valid_scene_ids.append(scene_idx[0])
        
        # Filter pairs to only include those where BOTH images are from valid scenes
        filtered_pairs = []
        for pair in self.pairs:
            im1_id = pair['im1_id']
            im2_id = pair['im2_id']
            
            # Check if both images belong to valid scenes
            im1_scene = self.images_scene_name[im1_id] if self.images_scene_name is not None else self.scenes[pair['scene_id']]
            im2_scene = self.images_scene_name[im2_id] if self.images_scene_name is not None else self.scenes[pair['scene_id']]
            
            if im1_scene in self.valid_scenes and im2_scene in self.valid_scenes:
                filtered_pairs.append(pair)
        
        self.pairs = filtered_pairs
        logging.info(f"Filtered to {len(self.pairs)} pairs from valid scenes")
        
        # Also update scene_to_images to only include valid scenes
        filtered_scene_to_images = {}
        for scene in self.valid_scenes:
            if scene in self.scene_to_images:
                filtered_scene_to_images[scene] = self.scene_to_images[scene]
        self.scene_to_images = filtered_scene_to_images

    def __len__(self):
        # return self.len_train
        return len(self.pairs)
        
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ):
        """
        Retrieve data for a specific sequence/pair.
        
        Returns:
            dict: Batch containing images, depths, camera parameters, etc.
        """
        # Random sampling if inside_random is True
        if self.inside_random:
            seq_index = random.randint(0, len(self.pairs) - 1) if len(self.pairs) > 0 else 0
        
        # Handle case where seq_index is larger than available pairs (for long training)
        if seq_index >= len(self.pairs):
            seq_index = seq_index % max(len(self.pairs), 1)
            
        # Get pair information
        if self.use_pairs and len(self.pairs) > 0:
            pair = self.pairs[seq_index]
            scene_id = pair['scene_id']
            im1_id = pair['im1_id']
            im2_id = pair['im2_id']
            
            # IMPORTANT: Use the scene name from images_scene_name, not from scene_id
            # because the scene_id in pairs might not correctly map to the scene
            scene_im1 = str(self.images_scene_name[im1_id]) if self.images_scene_name is not None else str(self.scenes[scene_id])
            scene_im2 = str(self.images_scene_name[im2_id]) if self.images_scene_name is not None else str(self.scenes[scene_id])
            
            # Verify both images are from the same scene
            if scene_im1 != scene_im2:
                logging.warning(f"Images from different scenes: {scene_im1} vs {scene_im2}, skipping pair")
                return self._get_dummy_batch(self.get_target_shape(aspect_ratio))
            
            scene = scene_im1  # Use the actual scene from the image mapping
            image_ids = [im1_id, im2_id]
            
            # If more images requested, sample from the same scene
            if img_per_seq > 2:
                scene_images = self.scene_to_images.get(scene, [])
                available_ids = [img_id for img_id, _ in scene_images 
                                if img_id not in image_ids]
                
                if len(available_ids) > 0 and self.get_nearby:
                    # Get nearby frames for better reconstruction
                    extra_needed = min(img_per_seq - 2, len(available_ids))
                    extra_ids = self.get_nearby_ids(
                        [im1_id], 
                        len(scene_images),
                        expand_ratio=self.expand_ratio
                    )[:extra_needed]
                    image_ids.extend(extra_ids)
                elif len(available_ids) > 0:
                    # Random sampling
                    extra_ids = np.random.choice(
                        available_ids, 
                        min(img_per_seq - 2, len(available_ids)),
                        replace=self.allow_duplicate_img
                    )
                    image_ids.extend(extra_ids)
        else:
            # Random sampling from a random scene
            scene = random.choice(self.valid_scenes) if self.valid_scenes else str(self.scenes[0])
            scene_images = self.scene_to_images.get(scene, [])
            
            if len(scene_images) >= img_per_seq:
                sampled = random.sample(scene_images, img_per_seq)
                image_ids = [img_id for img_id, _ in sampled]
            else:
                # Not enough images, use all available
                image_ids = [img_id for img_id, _ in scene_images]
                
        # Load and process images
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        
        successfully_loaded = 0
        for img_id in image_ids:
            try:
                img_name = str(self.images[img_id])
                
                # IMPORTANT: Get the correct scene for this specific image
                img_scene = str(self.images_scene_name[img_id]) if self.images_scene_name is not None else scene
                
                # Load image, depth, and camera parameters
                img_data = self._load_image_data(img_scene, img_name)
                if img_data is None:
                    continue
                    
                image, depth_map, cam2world, K = img_data
                
                # Convert cam2world to OpenCV world2cam (extrinsics)
                world2cam = np.linalg.inv(cam2world)
                extri_opencv = world2cam[:3, :]  # Take 3x4 part
                intri_opencv = K
                
                original_size = np.array(image.shape[:2])
                
                # Process image using base class method
                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    _,
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    filepath=osp.join(img_scene, img_name),
                )
                
                images.append(image)
                depths.append(depth_map)
                extrinsics.append(extri_opencv)
                intrinsics.append(intri_opencv)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                original_sizes.append(original_size)
                
                successfully_loaded += 1
                
            except Exception as e:
                logging.warning(f"Failed to load image {img_id}: {e}")
        
        # Ensure we have at least 2 images
        if successfully_loaded < 2:
            logging.warning(f"Only loaded {successfully_loaded} images, returning dummy data")
            # Return a dummy batch to avoid crashing
            return self._get_dummy_batch(target_image_shape)
            
        set_name = "aerial_megadepth"
        batch = {
            "seq_name": f"{set_name}_{scene}_{seq_index}",
            "ids": np.array(image_ids[:successfully_loaded]),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "tracks": None,  # Will be generated if needed
            "track_masks": None,
        }
        
        return batch
        
    def _load_image_data(self, scene, img_name):
        """Load image, depth, and camera parameters for a single frame."""
        scene_path = osp.join(self.ROOT, scene)
        
        try:
            # The actual image file has an additional .jpg appended during preprocessing
            img_path = osp.join(scene_path, img_name + '.jpg')
            
            if not osp.exists(img_path):
                logging.warning(f"Image not found: {img_path}")
                return None
                
            image = read_image_cv2(img_path, rgb=True)
            if image is None:
                return None
            
            # For depth and camera params, use the original name from train.npz
            # (without the extra .jpg that was added to the image file)
            depth_path = osp.join(scene_path, img_name + '.exr')
            if not osp.exists(depth_path):
                logging.warning(f"Depth not found: {depth_path}")
                return None
                
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth_map is None:
                return None
                
            if len(depth_map.shape) > 2:
                depth_map = depth_map[:, :, 0]
                
            # Load camera parameters
            npz_path = osp.join(scene_path, img_name + '.npz')
            if not osp.exists(npz_path):
                logging.warning(f"Camera params not found: {npz_path}")
                return None
                
            camera_params = np.load(npz_path)
            intrinsics = camera_params['intrinsics'].astype(np.float32)
            cam2world = camera_params['cam2world'].astype(np.float32)
            
            # Optional: Load and apply segmentation mask for sky removal
            if self.remove_sky and self.segmentation_root:
                seg_path = osp.join(self.segmentation_root, scene, img_name + '.png')
                if osp.exists(seg_path):
                    segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    # Remove sky (ADE20k label 2 is sky)
                    depth_map[segmap == 2] = 0
                    
            # Clean up depth map
            # Remove depths beyond max_depth (more aggressive for aerial)
            depth_map[depth_map > self.max_depth] = 0.0
            
            # Remove outliers using percentile
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 100:  # Need enough valid depths
                depth_threshold = np.percentile(valid_depths, self.depth_percentile)
                depth_map[depth_map > depth_threshold] = 0.0
                
                # Also remove very close depths (likely noise)
                min_threshold = np.percentile(valid_depths, 2)
                depth_map[depth_map < min_threshold] = 0.0
                
            return image, depth_map, cam2world, intrinsics
            
        except Exception as e:
            logging.warning(f"Failed to load {scene}/{img_name}: {e}")
            return None
    
    def _get_dummy_batch(self, target_shape):
        """Create a dummy batch for error cases."""
        h, w = target_shape
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_depth = np.zeros((h, w), dtype=np.float32)
        dummy_intrinsic = np.eye(3, dtype=np.float32)
        dummy_extrinsic = np.eye(3, 4, dtype=np.float32)
        dummy_points = np.zeros((h, w, 3), dtype=np.float32)
        dummy_mask = np.zeros((h, w), dtype=bool)
        
        return {
            "seq_name": "dummy_sequence",
            "ids": np.array([0, 1]),
            "frame_num": 2,
            "images": [dummy_img, dummy_img],
            "depths": [dummy_depth, dummy_depth],
            "extrinsics": [dummy_extrinsic, dummy_extrinsic],
            "intrinsics": [dummy_intrinsic, dummy_intrinsic],
            "cam_points": [dummy_points, dummy_points],
            "world_points": [dummy_points, dummy_points],
            "point_masks": [dummy_mask, dummy_mask],
            "original_sizes": [target_shape, target_shape],
            "tracks": None,
            "track_masks": None,
        }
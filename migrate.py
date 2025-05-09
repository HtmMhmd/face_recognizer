#!/usr/bin/env python3

"""
Migration script to help transition from the old project structure to the new organized structure.
This script will:
1. Create missing directories
2. Copy files to their new locations (not deleting originals yet)
3. Fix import statements in the new files
"""

import os
import shutil
import re
import sys
from pathlib import Path

# Define source and destination mappings
MIGRATIONS = [
    # Format: (source_path, destination_path, module_name)
    # Core components
    ('Verify/Verify.py', 'src/core/verification/verify.py', 'FaceVerifier'),
    ('Verify/verify_utilis.py', 'src/core/verification/utils.py', None),
    ('Align/Align.py', 'src/core/alignment/align.py', 'FaceAligner'),
    ('Align/EyeDetect.py', 'src/core/alignment/eye_detect.py', 'EyeDetector'),
    ('drowsiness/EAR.py', 'src/core/drowsiness/detection.py', 'DrowsinessDetector'),
    
    # Model components
    ('Model/Detector.py', 'src/models/detection/detector.py', 'Detector'),
    ('Model/FaceDetection.py', 'src/models/detection/base.py', 'FaceDetector'),
    ('Model/DetectionFaces.py', 'src/models/detection/detection_faces.py', 'DetectionFaces'),
    ('Model/DetectionResult.py', 'src/models/detection/detection_result.py', 'DetectionResult'),
    ('Model/DetectionEmbedding.py', 'src/models/detection/detection_embedding.py', 'DetectionEmbedding'),
    ('Model/detection_utilis.py', 'src/models/detection/utils.py', None),
    ('Model/FacialRecognition.py', 'src/models/face_recognition/base.py', 'FacialRecognition'),
    ('Model/FaceNet/FaceNetTFLiteHandler.py', 'src/models/face_recognition/facenet.py', 'FaceNetTFLiteHandler'),
    ('Model/MediapipeDetection/MediapipeFaceDetector.py', 'src/models/detection/mediapipe_detector.py', 'MediapipeFaceDetector'),
    ('Model/MediapipeDetection/MediapipeFaceLandmarker.py', 'src/models/detection/mediapipe_landmarker.py', 'FaceMeshDetector'),
    ('Model/YoloDetection/Yolov8OnnxRuntimeDetector.py', 'src/models/detection/yolo_detector.py', 'Yolov8OnnxRuntimeDetector'),
    
    # Utils
    ('CameraUtilis/CameraHandler.py', 'src/utils/camera/camera_handler.py', 'CameraHandler'),
    ('CameraUtilis/MultipleCameras.py', 'src/utils/camera/multiple_cameras.py', None),
    ('ImageUtilis/image_utilis.py', 'src/utils/image/image_processing.py', 'preprocess_image'),
    
    # Database
    ('database/db_api.py', 'src/database/api/db_api.py', None),
    ('database/db_client.py', 'src/database/api/db_client.py', None),
    ('database/db_handler.py', 'src/database/handlers/db_handler.py', None),
    ('UsersDatabaseHandeler/UsersDatabaseHandeler.py', 'src/database/handlers/user_db_handler.py', 'UsersDatabaseHandeler'),
    
    # API
    ('api.py', 'src/api/app.py', None),
    ('templates/index.html', 'src/api/templates/index.html', None),
    ('templates/video_feed.html', 'src/api/templates/video_feed.html', None),
]

def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    
    # Create an __init__.py file if it doesn't exist
    init_file = os.path.join(directory, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Automatically created by migration script\n')

def copy_file(source, destination):
    """Copy a file, creating the destination directory if necessary."""
    dest_dir = os.path.dirname(destination)
    ensure_dir(dest_dir)
    
    if os.path.exists(source):
        print(f"Copying {source} to {destination}")
        shutil.copy2(source, destination)
    else:
        print(f"Warning: Source file {source} does not exist")

def update_imports(file_path, old_module_prefix, new_module_prefix):
    """Update import statements in a file to use the new module structure."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace direct imports
    updated_content = re.sub(
        r'from\s+{0}\s+import'.format(re.escape(old_module_prefix)), 
        f'from {new_module_prefix} import', 
        content
    )
    
    # Replace import statements
    updated_content = re.sub(
        r'import\s+{0}'.format(re.escape(old_module_prefix)), 
        f'import {new_module_prefix}', 
        updated_content
    )
    
    with open(file_path, 'w') as f:
        f.write(updated_content)

def create_init_files():
    """Create __init__.py files with appropriate imports."""
    init_files = {
        'src/models/detection/__init__.py': [
            'from .detector import Detector',
            'from .base import FaceDetector',
            'from .detection_faces import DetectionFaces',
            'from .detection_result import DetectionResult',
            'from .detection_embedding import DetectionEmbedding',
            'from .mediapipe_detector import MediapipeFaceDetector',
            'from .mediapipe_landmarker import FaceMeshDetector',
            'from .yolo_detector import Yolov8OnnxRuntimeDetector',
            '',
            '__all__ = [',
            '    "Detector",',
            '    "FaceDetector",',
            '    "DetectionFaces",',
            '    "DetectionResult",',
            '    "DetectionEmbedding",',
            '    "MediapipeFaceDetector",',
            '    "FaceMeshDetector",',
            '    "Yolov8OnnxRuntimeDetector",',
            ']',
        ],
        'src/models/face_recognition/__init__.py': [
            'from .base import FacialRecognition',
            'from .facenet import FaceNetTFLiteHandler',
            '',
            '__all__ = [',
            '    "FacialRecognition",',
            '    "FaceNetTFLiteHandler",',
            ']',
        ],
        'src/core/alignment/__init__.py': [
            'from .align import FaceAligner',
            'from .eye_detect import EyeDetector',
            '',
            '__all__ = [',
            '    "FaceAligner",',
            '    "EyeDetector",',
            ']',
        ],
        'src/utils/image/__init__.py': [
            'from .image_processing import preprocess_image, resize_image',
            '',
            '__all__ = [',
            '    "preprocess_image",',
            '    "resize_image",',
            ']',
        ],
        'src/database/__init__.py': [
            'from .face_db import FaceDatabase',
            '',
            '__all__ = [',
            '    "FaceDatabase",',
            ']',
        ],
        'src/database/face_db.py': [
            'from .handlers.user_db_handler import UsersDatabaseHandeler',
            '',
            'class FaceDatabase:',
            '    """',
            '    A wrapper class for face database operations.',
            '    """',
            '    def __init__(self):',
            '        self.db_handler = UsersDatabaseHandeler()',
            '',
            '    def get_all_embeddings(self):',
            '        """Get all user embeddings from the database."""',
            '        return self.db_handler.get_all_embeddings()',
            '',
            '    def add_or_update_user(self, username, embedding):',
            '        """Add or update a user in the database."""',
            '        return self.db_handler.add_or_update_user(username, embedding)',
            '',
            '    def delete_user(self, username):',
            '        """Delete a user from the database."""',
            '        return self.db_handler.delete_user(username)',
            '',
            '    def update_last_login(self, username):',
            '        """Update the last login time for a user."""',
            '        return self.db_handler.update_last_login(username)',
        ],
        'src/utils/image/drawing.py': [
            'import cv2',
            'import numpy as np',
            '',
            'def draw_user_names_on_bboxes(image, results):',
            '    """',
            '    Draw user names on bounding boxes.',
            '    ',
            '    Args:',
            '        image (np.ndarray): Input image',
            '        results (list): List of dictionaries with bbox, user_name, and verification_result keys',
            '        ',
            '    Returns:',
            '        np.ndarray: Image with user names drawn on bounding boxes',
            '    """',
            '    if not results:',
            '        return image',
            '        ',
            '    img_with_names = image.copy()',
            '    ',
            '    for result in results:',
            '        bbox = result.get("bbox")',
            '        user_name = result.get("user_name", "Unknown")',
            '        ',
            '        if bbox is None:',
            '            continue',
            '            ',
            '        # Draw rectangle and name',
            '        cv2.rectangle(img_with_names, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)',
            '        cv2.putText(img_with_names, user_name, (bbox[0], bbox[1] - 10),',
            '                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)',
            '                   ',
            '    return img_with_names',
        ],
    }
    
    for file_path, content in init_files.items():
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Created {file_path}")

def main():
    """Main migration function."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure all necessary directories exist
    for _, dest, _ in MIGRATIONS:
        dest_dir = os.path.dirname(os.path.join(base_dir, dest))
        ensure_dir(dest_dir)
    
    # Copy files to their new locations
    for source, dest, _ in MIGRATIONS:
        source_path = os.path.join(base_dir, source)
        dest_path = os.path.join(base_dir, dest)
        copy_file(source_path, dest_path)
    
    # Create initialization files
    create_init_files()
    
    print("\nMigration completed successfully!")
    print("Please review the new file structure and make any necessary adjustments.")
    print("\nTo switch to the new structure completely:")
    print("1. Test the application using the new entry points (run.py)")
    print("2. If everything works, you can remove the original files.")
    print("3. Make sure to update your Docker and deployment configurations.")

if __name__ == "__main__":
    main()
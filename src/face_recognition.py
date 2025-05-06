import cv2
import logging
import pickle
import numpy as np
import time
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from database_handler import DatabaseHandler

# Setup logging
logger = logging.getLogger("AttendanceSystem")

class AttendanceSystem:
    def __init__(self, config, db_handler: Optional[DatabaseHandler] = None):
        self.config = config
        self.last_attendance = {}  # To prevent multiple attendances in a short time
        
        # Use existing DB handler if provided; else create new
        if db_handler is not None:
            self.db = db_handler
        else:
            db_path = config.get("db_path", "attendance.db")
            self.db = DatabaseHandler(db_path)
        
        # Initialize face embedding module with SCRFD and FaceNet512B
        from face_embedding import FaceEmbedding, dict_to_face_embedding_config
    
        # Convert config dict to FaceEmbeddingConfig to ensure data type safety
        self.face_embedding = FaceEmbedding(dict_to_face_embedding_config(self.config))
        
        # Load embeddings
        self.update_embeddings()
        
        logger.debug("Attendance System initialized with SCRFD 2.5G and FaceNet512B")

    def update_embeddings(self) -> None:
        """Update embeddings from local DB"""
        embeddings = self.db.get_all_embeddings()
        self.face_embedding.load_embeddings(embeddings)
        logger.debug("Embeddings updated")

    def record_attendance(self, student_id: str, student_name: str, confidence: float, force: bool = False) -> bool:
        """Record attendance with time check to prevent duplicates"""
        current_time = time.time()
        cooldown = self.config.get("attendance_cooldown", 300)  # Default 5 minutes
        
        # Only record if at least cooldown seconds since last attendance, unless forced (manual attendance)
        if force or student_id not in self.last_attendance or current_time - self.last_attendance[student_id] > cooldown:
            # Get class_name from DB efficiently
            class_name = self.db.get_class_by_student_id(student_id)
            status = "present"
            
            # Record attendance
            if self.db.record_attendance(student_id, student_name, class_name, status, confidence):
                self.last_attendance[student_id] = current_time
                logger.debug(f"Recorded attendance for student {student_name} with confidence {confidence}")
                return True
        return False

    def process_image(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process image and record attendance using SCRFD and FaceNet512B"""
        # Use face_embedding module to process image
        img, recognized_faces = self.face_embedding.process_image(image)
        
        # Log hash of input image before passing to embedding (if ndarray)
        if isinstance(image, np.ndarray):
            logger.debug(f"[Realtime] Input image hash: {hash(image.tobytes())}")
            
        # Create a copy of the image for drawing
        result_img = img.copy()
        
        # Process recognized faces
        recog_threshold = 1 - self.config["recognition_distance"]
        for face in recognized_faces:
            # Record attendance if confidence above threshold
            if face["student_id"] and face["confidence"] > recog_threshold:
                face["attendance_recorded"] = self.record_attendance(
                    face["student_id"], face["name"], face["confidence"]
                )
            else:
                face["attendance_recorded"] = False
        
        # Draw results on image
        result_img = self.draw_faces_on_image(result_img, recognized_faces)
        
        return result_img, recognized_faces
        
    def add_student(self, name: str, face_image: np.ndarray) -> Optional[str]:
        """Add a new student with face image using FaceNet512B embeddings"""
        try:
            # Extract face embedding with FaceNet512B
            embedding = self.face_embedding.get_face_embedding(face_image)
            if embedding is None:
                logger.error(f"Could not extract embedding for student {name}")
                return None
            
            # Add to database
            student_id = self.db.add_student(name, embedding)
            if student_id:
                # Update embeddings
                self.update_embeddings()
                logger.debug(f"Added student {name} with ID {student_id}")
            
            return student_id
        except Exception as e:
            logger.error(f"Error adding student {name}: {str(e)}")
            return None
    
    def process_student_image(self, student_id: str, image_path: str) -> bool:
        """Update student face image using SCRFD detection & FaceNet512B"""
        try:
            # Extract face from image using SCRFD
            face_roi = self.face_embedding.extract_face_from_image(image_path)
            if face_roi is None:
                return False
                
            # Update student with FaceNet512B embedding
            embedding = self.face_embedding.get_face_embedding(face_roi)
            if embedding is None:
                logger.error(f"Could not extract embedding from image {image_path}")
                return False
            
            # Update in database
            success = self.db.update_student(student_id, {"embedding": pickle.dumps(embedding)})
            if success:
                # Update embeddings
                self.update_embeddings()
                logger.debug(f"Updated face image for student {student_id}")
            
            return success
        except Exception as e:
            logger.error(f"Error processing student image: {str(e)}")
            return False
    
    def save_attendance_report(self, path: Optional[str] = None) -> Optional[str]:
        """Save attendance report for today"""
        if path is None:
            now = datetime.now().strftime("%Y%m%d")
            path = os.path.join(self.config["result_dir"], f"attendance_{now}.csv")
            
        try:
            attendance_records = self.db.get_attendance_by_date()
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write("No.,Name,Time,Confidence\n")
                for i, (_, name, time, confidence) in enumerate(attendance_records, 1):
                    f.write(f"{i},{name},{time},{confidence:.2f}\n")
                    
            logger.debug(f"Saved attendance report at: {path}")
            
            return path
        except Exception as e:
            logger.error(f"Error saving attendance report: {str(e)}")
            return None
    
    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel and record attendance"""
        results = []
        for result in self.face_embedding.batch_process_images(image_paths):
            if result["success"]:
                # Process for attendance
                processed_img = result["processed_image"].copy()
                recognized = result["faces"]
                
                # Record attendance for each face
                for face in recognized:
                    student_id = face.get("student_id")
                    name = face.get("name")
                    confidence = face.get("confidence", 0)
                    
                    if student_id and confidence > self.config["recognition_confidence"]:
                        face["attendance_recorded"] = self.record_attendance(student_id, name, confidence)
                    else:
                        face["attendance_recorded"] = False
                
                # Update result
                result["processed_image"] = self.draw_faces_on_image(processed_img, recognized)
            
            results.append(result)
            
        return results

    def draw_faces_on_image(self, img: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw faces, names and attendance status on an image"""
        result_img = img.copy()
        for face in faces:
            if "bbox" not in face:
                continue
                
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            confidence = face["confidence"]
            attendance_status = face.get("attendance_recorded", False)
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            # Different color if attendance was recorded
            if attendance_status:
                color = (255, 0, 0)  # Red when attendance recorded
                
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Display name and status
            status = "✓" if attendance_status else ""
            label = f"{name} ({confidence:.2f}) {status}"
            
            # Ensure text is displayed within frame
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            
            # Draw black background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, text_y - 15), (x1 + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display current information
        current_time = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
        
        # Draw black background for timestamp
        text_size = cv2.getTextSize(f"Time: {current_time}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (10, 5), (10 + text_size[0], 30), (0, 0, 0), -1)
        
        cv2.putText(result_img, f"Time: {current_time}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_img
    
    def delete_student(self, student_id: str) -> bool:
        """Delete a student from the system"""
        success = self.db.delete_student(student_id)
        if success:
            # Update embeddings
            self.update_embeddings()
            logger.debug(f"Deleted student {student_id}")
        return success
        
    def close(self) -> None:
        """Close resources"""
        self.face_embedding.close()
        self.db.close()
        logger.debug("Attendance System closed")
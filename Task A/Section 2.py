import cv2
import numpy as np
import os

"""Blur all the faces (camera facing) that appear in a video."""

VIDEO_FOLDER = "Recorded Videos (4)/"
OUTPUT_FOLDER = "Output/Task A/Section 2/"
FACE_CASCADE_PATH = "Dependency/face_detector.xml"

# 自动适配KCF或CSRT追踪器
def create_tracker():
    if hasattr(cv2, 'legacy'):
        # 新版OpenCV (>=4.5.2)
        if hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    else:
        # 旧版OpenCV
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
    raise AttributeError("No supported tracker found! Please install opencv-contrib-python.")

class FaceTracker:
    def __init__(self, bbox, frame, tolerance_frames=15):
        self.tracker = create_tracker()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        self.lost_frames = 0
        self.tolerance_frames = tolerance_frames  # 容许丢失的帧数
        self.active = True
        self.confidence_history = []  # 追踪置信度历史
        self.max_history = 10
        self.expand_ratio = 1.15  # 扩展模糊区域的比例
        
        # 平滑追踪相关
        self.bbox_history = [bbox]  # 位置历史
        self.velocity = (0, 0)  # 速度向量 (vx, vy)
        self.max_bbox_history = 5
        self.smooth_factor = 0.3  # 平滑因子
        
    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        if ok:
            # 平滑处理新的边界框
            smoothed_bbox = self.smooth_bbox(bbox)
            
            # 更新置信度历史
            self.confidence_history.append(1.0)
            if len(self.confidence_history) > self.max_history:
                self.confidence_history.pop(0)
            
            # 更新位置历史和速度
            self.update_motion_model(smoothed_bbox)
            self.last_bbox = smoothed_bbox
            self.lost_frames = 0
            return True, smoothed_bbox
        else:
            # 记录失败的追踪
            self.confidence_history.append(0.0)
            if len(self.confidence_history) > self.max_history:
                self.confidence_history.pop(0)
                
            self.lost_frames += 1
            if self.lost_frames > self.tolerance_frames:
                self.active = False
                return False, None
            
            # 在容许时间内，使用运动预测的位置
            predicted_bbox = self.predict_next_position()
            return True, predicted_bbox
    
    def smooth_bbox(self, new_bbox):
        """对边界框进行平滑处理，减少抖动"""
        if len(self.bbox_history) == 0:
            return new_bbox
        
        last_bbox = self.bbox_history[-1]
        x1, y1, w1, h1 = last_bbox
        x2, y2, w2, h2 = new_bbox
        
        # 使用加权平均进行平滑
        smooth_x = x1 * (1 - self.smooth_factor) + x2 * self.smooth_factor
        smooth_y = y1 * (1 - self.smooth_factor) + y2 * self.smooth_factor
        smooth_w = w1 * (1 - self.smooth_factor) + w2 * self.smooth_factor
        smooth_h = h1 * (1 - self.smooth_factor) + h2 * self.smooth_factor
        
        return (smooth_x, smooth_y, smooth_w, smooth_h)
    
    def update_motion_model(self, bbox):
        """更新运动模型，计算速度向量"""
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.max_bbox_history:
            self.bbox_history.pop(0)
        
        # 计算速度向量（基于最近两个位置）
        if len(self.bbox_history) >= 2:
            curr_x, curr_y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            prev_x, prev_y = (self.bbox_history[-2][0] + self.bbox_history[-2][2]/2, 
                             self.bbox_history[-2][1] + self.bbox_history[-2][3]/2)
            
            # 更新速度向量（使用平滑）
            new_vx = curr_x - prev_x
            new_vy = curr_y - prev_y
            self.velocity = (self.velocity[0] * 0.7 + new_vx * 0.3,
                           self.velocity[1] * 0.7 + new_vy * 0.3)
    
    def predict_next_position(self):
        """基于运动模型预测下一个位置"""
        if not self.bbox_history:
            return self.get_expanded_bbox()
        
        last_bbox = self.bbox_history[-1]
        x, y, w, h = last_bbox
        
        # 使用速度向量预测新位置
        predicted_x = x + self.velocity[0] * self.lost_frames
        predicted_y = y + self.velocity[1] * self.lost_frames
        
        # 考虑速度衰减（人不会一直匀速运动）
        decay_factor = 0.95 ** self.lost_frames
        predicted_x = x + (predicted_x - x) * decay_factor
        predicted_y = y + (predicted_y - y) * decay_factor
        
        # 保持固定的扩展比例，不随时间增大
        expand_ratio = 1.15  # 固定扩展比例
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        expand_x = predicted_x - (expand_w - w) / 2
        expand_y = predicted_y - (expand_h - h) / 2
        
        return (expand_x, expand_y, expand_w, expand_h)

    def get_expanded_bbox(self):
        """获取扩展的边界框，用于处理侧脸或部分遮挡"""
        x, y, w, h = self.last_bbox
        # 扩展边界框
        expand_w = int(w * self.expand_ratio)
        expand_h = int(h * self.expand_ratio)
        expand_x = int(x - (expand_w - w) / 2)
        expand_y = int(y - (expand_h - h) / 2)
        return (expand_x, expand_y, expand_w, expand_h)
    
    def get_confidence(self):
        """获取追踪器的平均置信度"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

def blur_faces_in_frame(frame, face_trackers):
    for tracker in face_trackers[:]:  # 使用副本遍历，避免修改列表时出错
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(i) for i in bbox]
            # 边界处理，防止越界
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            
            # 根据追踪状态调整模糊强度
            blur_intensity = get_adaptive_blur_intensity(tracker)
            
            # 按比例动态调整模糊核
            base_blur = max(21, w//3*2+1)
            blur_size = (base_blur, base_blur)  # 保证为奇数
            sigma = 30 * blur_intensity
            
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_roi = cv2.GaussianBlur(face_roi, blur_size, sigma)
                frame[y:y+h, x:x+w] = face_roi
        elif not tracker.active:
            # 移除不活跃的追踪器
            face_trackers.remove(tracker)
    return frame

def get_adaptive_blur_intensity(tracker):
    """根据追踪器状态自适应调整模糊强度"""
    confidence = tracker.get_confidence()
    lost_frames = tracker.lost_frames
    
    # 基础强度
    base_intensity = 1.0
    
    # 置信度越低，模糊稍微增强
    confidence_factor = 1.0 + (1.0 - confidence) * 0.2  # 减少影响
    
    # 丢失帧数的影响：保持模糊强度相对稳定
    if lost_frames > 0:
        # 轻微减弱模糊，但不会太明显
        fade_factor = max(0.8, 1.0 - lost_frames * 0.02)  # 更温和的衰减
    else:
        fade_factor = 1.0
    
    return base_intensity * confidence_factor * fade_factor

def validate_face_region(frame, bbox):
    """验证检测到的区域是否真的是人脸"""
    x, y, w, h = [int(v) for v in bbox]
    
    # 确保边界框在图像范围内
    x = max(0, min(x, frame.shape[1] - 1))
    y = max(0, min(y, frame.shape[0] - 1))
    w = max(1, min(w, frame.shape[1] - x))
    h = max(1, min(h, frame.shape[0] - y))
    
    # 提取候选区域
    face_region = frame[y:y+h, x:x+w]
    if face_region.size == 0:
        return False
    
    # 转换为灰度图
    if len(face_region.shape) == 3:
        gray_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = face_region
    
    # 1. 检查区域的纹理复杂度（人脸通常有适度的纹理）
    # 使用Laplacian算子检测边缘
    laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
    texture_variance = laplacian.var()
    
    # 人脸通常有适度的纹理复杂度（不会太平滑也不会太复杂）
    if texture_variance < 50 or texture_variance > 8000:
        return False
    
    # 2. 检查亮度分布（人脸通常有相对均匀的亮度分布）
    mean_brightness = gray_region.mean()
    brightness_std = gray_region.std()
    
    # 排除过暗、过亮或亮度变化过大的区域
    if mean_brightness < 20 or mean_brightness > 220 or brightness_std > 80:
        return False
    
    # 3. 检查长宽比（已在detect_faces中检查，这里再次确认）
    aspect_ratio = w / h
    if not (0.6 <= aspect_ratio <= 1.6):
        return False
    
    return True

def classify_scene_density(faces):
    """根据检测到的人脸数量分类场景密度"""
    face_count = len(faces)
    if face_count <= 2:        # 更保守的阈值
        return "low"  # 低密度：1-2个人脸
    elif face_count <= 5:      # 更保守的阈值
        return "medium"  # 中密度：3-5个人脸
    else:
        return "high"  # 高密度：6个以上人脸

def merge_overlapping_faces(faces, threshold=0.4):
    """合并重叠的人脸检测框，避免过度分割"""
    if len(faces) <= 1:
        return faces
    
    merged = []
    used = [False] * len(faces)
    
    for i, face1 in enumerate(faces):
        if used[i]:
            continue
            
        x1, y1, w1, h1 = face1
        merged_face = [x1, y1, x1+w1, y1+h1]  # 转换为 [x1, y1, x2, y2] 格式
        
        for j, face2 in enumerate(faces[i+1:], i+1):
            if used[j]:
                continue
                
            x2, y2, w2, h2 = face2
            # 计算重叠度
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                if intersection / union > threshold:
                    # 合并边界框
                    merged_face[0] = min(merged_face[0], x2)
                    merged_face[1] = min(merged_face[1], y2)
                    merged_face[2] = max(merged_face[2], x2 + w2)
                    merged_face[3] = max(merged_face[3], y2 + h2)
                    used[j] = True
        
        # 转换回 [x, y, w, h] 格式
        merged.append((merged_face[0], merged_face[1], 
                      merged_face[2] - merged_face[0], 
                      merged_face[3] - merged_face[1]))
        used[i] = True
    
    return merged

def adaptive_blur_region(frame, bbox, density="low", tracker=None):
    """根据场景密度自适应调整模糊区域"""
    x, y, w, h = [int(v) for v in bbox]
    
    # 根据密度调整扩展比例（固定比例，不随时间变化）
    if density == "high":
        expand_ratio = 1.25  # 高密度时扩大模糊区域
    elif density == "medium":
        expand_ratio = 1.15
    else:
        expand_ratio = 1.1   # 低密度时精确模糊
    
    # 扩展边界框
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    expand_x = max(0, int(x - (expand_w - w) / 2))
    expand_y = max(0, int(y - (expand_h - h) / 2))
    
    # 确保不越界
    expand_x = max(0, min(expand_x, frame.shape[1] - 1))
    expand_y = max(0, min(expand_y, frame.shape[0] - 1))
    expand_w = max(1, min(expand_w, frame.shape[1] - expand_x))
    expand_h = max(1, min(expand_h, frame.shape[0] - expand_y))
    
    # 获取自适应模糊强度（保持相对稳定）
    if tracker:
        blur_intensity = get_adaptive_blur_intensity(tracker)
    else:
        blur_intensity = 1.0
    
    # 自适应模糊强度
    if density == "high":
        blur_kernel = (max(31, expand_w//4*2+1), max(31, expand_h//4*2+1))
        sigma = max(20, 40 * blur_intensity)  # 确保最小模糊强度
    else:
        blur_kernel = (max(21, expand_w//3*2+1), max(21, expand_h//3*2+1))
        sigma = max(15, 30 * blur_intensity)  # 确保最小模糊强度
    
    face_roi = frame[expand_y:expand_y+expand_h, expand_x:expand_x+expand_w]
    if face_roi.size > 0:
        face_roi = cv2.GaussianBlur(face_roi, blur_kernel, sigma)
        frame[expand_y:expand_y+expand_h, expand_x:expand_x+expand_w] = face_roi
    
    return frame

def is_face_overlap(bbox1, bbox2, threshold=0.3):
    """检查两个面部区域是否重叠"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area > threshold

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用更严格的参数减少误判
    # 主要检测器：使用较严格的参数
    faces_main = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.15,      # 更保守的缩放因子
        minNeighbors=6,        # 增加最小邻居数，减少误判
        minSize=(40, 40),      # 最小人脸尺寸
        maxSize=(250, 250),    # 限制最大尺寸，避免检测建筑物
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 补充检测器：用于捕获可能遗漏的小人脸
    faces_small = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(120, 120),    # 只检测小人脸
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 合并检测结果
    all_faces = list(faces_main) + list(faces_small)
    if len(all_faces) == 0:
        return []
    
    # 去除重复检测，使用更严格的阈值
    merged_faces = merge_overlapping_faces(all_faces, threshold=0.4)
    
    # 进一步过滤：基于长宽比和大小过滤可能的误检
    filtered_faces = []
    for (x, y, w, h) in merged_faces:
        # 人脸长宽比通常在0.7-1.4之间
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.6:
            # 过滤过大的检测框（可能是建筑物）
            face_area = w * h
            frame_area = gray.shape[0] * gray.shape[1]
            if face_area / frame_area < 0.25:  # 人脸不应该占画面的25%以上
                filtered_faces.append((x, y, w, h))
    
    return filtered_faces

def process_video(input_path, output_path, face_cascade):
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        print(f"Failed to open video: {input_path}")
        return
    
    # 自动选择合适的编码和扩展名
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.avi']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    face_trackers = []
    tolerance_frames = max(20, fps // 2)  # 增加容许时间，至少20帧
    scene_density = "low"  # 场景密度
    high_density_mode = False  # 高密度模式标志
    last_detection_faces_count = 0
    
    while True:
        success, frame = vid.read()
        if not success:
            break

        # 自适应检测间隔
        if high_density_mode:
            detection_interval = 12  # 高密度场景适度频繁检测
        else:
            detection_interval = 18  # 正常场景减少检测频率

        # 定期重新检测人脸，或者当没有活跃追踪器时
        active_trackers = [t for t in face_trackers if t.active]
        should_detect = (frame_id % detection_interval == 0 or 
                        len(active_trackers) == 0 or
                        len(active_trackers) < last_detection_faces_count * 0.7)  # 如果追踪器数量明显减少
        
        if should_detect:
            faces = detect_faces(frame, face_cascade)
            
            # 验证检测到的人脸，过滤误检
            validated_faces = []
            for face in faces:
                if validate_face_region(frame, face):
                    validated_faces.append(face)
            
            faces = validated_faces
            last_detection_faces_count = len(faces)
            scene_density = classify_scene_density(faces)
            
            # 启用高密度模式（阈值降低）
            if scene_density == "high":
                high_density_mode = True
                tolerance_frames = max(25, fps // 1.8)  # 高密度时增加容许时间
            else:
                high_density_mode = False
                tolerance_frames = max(20, fps // 2)
            
            # 对于检测到的每个人脸，检查是否已经有追踪器在跟踪
            for (x, y, w, h) in faces:
                new_bbox = (x, y, w, h)
                is_new_face = True
                overlap_threshold = 0.35 if scene_density == "high" else 0.25  # 更严格的重叠阈值
                
                # 检查是否与现有追踪器重叠
                for tracker in face_trackers:
                    if tracker.active and is_face_overlap(new_bbox, tracker.last_bbox, overlap_threshold):
                        is_new_face = False
                        # 如果追踪器置信度较低，重新初始化
                        if tracker.get_confidence() < 0.7:  # 提高置信度阈值
                            tracker.tracker = create_tracker()
                            tracker.tracker.init(frame, new_bbox)
                            # 保持运动历史的连续性
                            tracker.update_motion_model(new_bbox)
                            tracker.lost_frames = 0
                        break
                
                # 如果是新人脸，创建新的追踪器
                if is_new_face:
                    new_tracker = FaceTracker(new_bbox, frame, tolerance_frames)
                    face_trackers.append(new_tracker)

        # 根据场景密度选择模糊策略
        if scene_density == "high":
            # 高密度场景：使用自适应模糊
            for tracker in face_trackers[:]:
                ok, bbox = tracker.update(frame)
                if ok:
                    frame = adaptive_blur_region(frame, bbox, scene_density, tracker)
                elif not tracker.active:
                    face_trackers.remove(tracker)
        else:
            # 低/中密度场景：使用原有模糊方式
            frame = blur_faces_in_frame(frame, face_trackers)
            
        out.write(frame)
        frame_id += 1
        
        # 进度提示
        if frame_id % 100 == 0:
            active_count = len([t for t in face_trackers if t.active])
            print(f"{input_path}: {frame_id} frames processed, {active_count} active trackers, density: {scene_density}")

    vid.release()
    out.release()
    print(f"Finished processing: {input_path}")
    print(f"Total frames processed: {frame_id}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    for filename in os.listdir(VIDEO_FOLDER):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(VIDEO_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"blurred_{filename}")
            process_video(input_path, output_path, face_cascade)

if __name__ == "__main__":
    main()

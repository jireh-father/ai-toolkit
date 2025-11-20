import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import mediapipe as mp


def calculate_face_similarity(face1, face2):
    """두 얼굴의 임베딩 유사도 계산 (코사인 유사도)"""
    embedding1 = face1.normed_embedding
    embedding2 = face2.normed_embedding
    
    # 코사인 유사도 계산
    similarity = np.dot(embedding1, embedding2)
    return float(similarity)


def check_face_keypoints_displacement(face1, face2, image_size, displacement_threshold=0.1):
    """얼굴 키포인트 변위 검사 (이미지 크기 대비 비율)"""
    kps1 = face1.kps  # shape: (5, 2) - left_eye, right_eye, nose, left_mouth, right_mouth
    kps2 = face2.kps
    
    # 이미지 대각선 길이
    diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
    
    # 각 키포인트의 변위 계산
    for i in range(len(kps1)):
        displacement = np.linalg.norm(kps1[i] - kps2[i])
        displacement_ratio = displacement / diagonal
        
        if displacement_ratio > displacement_threshold:
            return False, displacement_ratio
    
    return True, 0.0


def check_mediapipe_pose_displacement(image1, image2, displacement_threshold=0.1):
    """MediaPipe Pose로 신체 키포인트 변위 검사"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    # 이미지를 RGB로 변환
    image1_rgb = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image2_rgb = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    
    # Pose 검출
    results1 = pose.process(cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB))
    results2 = pose.process(cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB))
    
    pose.close()
    
    # 두 이미지 모두에서 pose가 검출되지 않으면 통과
    if not results1.pose_landmarks or not results2.pose_landmarks:
        return True, 0.0
    
    # 이미지 대각선 길이
    height, width = image1_rgb.shape[:2]
    diagonal = np.sqrt(width**2 + height**2)
    
    landmarks1 = results1.pose_landmarks.landmark
    landmarks2 = results2.pose_landmarks.landmark
    
    # 각 랜드마크의 변위 계산
    max_displacement_ratio = 0.0
    for lm1, lm2 in zip(landmarks1, landmarks2):
        # 픽셀 좌표로 변환
        x1, y1 = lm1.x * width, lm1.y * height
        x2, y2 = lm2.x * width, lm2.y * height
        
        # 변위 계산
        displacement = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        displacement_ratio = displacement / diagonal
        
        max_displacement_ratio = max(max_displacement_ratio, displacement_ratio)
        
        if displacement_ratio > displacement_threshold:
            return False, displacement_ratio
    
    return True, max_displacement_ratio


def validate_image_pair(input_path, output_path, face_app, face_similarity_threshold, 
                        face_keypoint_threshold, pose_threshold):
    """이미지 쌍 검증"""
    # 이미지 로드
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)
    
    # RGB로 변환
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    if output_image.mode != 'RGB':
        output_image = output_image.convert('RGB')
    
    # numpy array로 변환
    input_np = np.array(input_image)
    output_np = np.array(output_image)
    
    # InsightFace로 얼굴 검출
    faces_input = face_app.get(input_np)
    faces_output = face_app.get(output_np)
    
    if len(faces_input) == 0 or len(faces_output) == 0:
        return False, "얼굴이 검출되지 않음"
    
    # 가장 큰 얼굴 사용
    face_input = max(faces_input, key=lambda x: x.bbox[2] * x.bbox[3])
    face_output = max(faces_output, key=lambda x: x.bbox[2] * x.bbox[3])
    
    # 1. 얼굴 유사도 검사 (threshold 이하여야 성공)
    similarity = calculate_face_similarity(face_input, face_output)
    if similarity < face_similarity_threshold:
        return False, f"얼굴 유사도가 너무 낮음: {similarity:.4f} < {face_similarity_threshold}"
    
    # 2. 얼굴 키포인트 변위 검사
    keypoint_ok, displacement = check_face_keypoints_displacement(
        face_input, face_output, 
        (input_image.width, input_image.height),
        face_keypoint_threshold
    )
    if not keypoint_ok:
        return False, f"얼굴 키포인트 변위 초과: {displacement:.4f} > {face_keypoint_threshold}"
    
    # 3. MediaPipe Pose 키포인트 변위 검사
    pose_ok, pose_displacement = check_mediapipe_pose_displacement(
        input_image, output_image, pose_threshold
    )
    if not pose_ok:
        return False, f"Pose 키포인트 변위 초과: {pose_displacement:.4f} > {pose_threshold}"
    
    return True, "검증 성공"


def main():
    parser = argparse.ArgumentParser(description="이미지 편집 데이터셋 검증")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 이미지 디렉토리")
    parser.add_argument("--false_dir", type=str, default="false_cases", help="실패 케이스 저장 디렉토리")
    parser.add_argument("--face_similarity_threshold", type=float, default=0.4, 
                       help="얼굴 유사도 임계값 (이하여야 성공)")
    parser.add_argument("--face_keypoint_threshold", type=float, default=0.1,
                       help="얼굴 키포인트 변위 임계값 (이미지 대각선 대비 비율)")
    parser.add_argument("--pose_threshold", type=float, default=0.1,
                       help="Pose 키포인트 변위 임계값 (이미지 대각선 대비 비율)")
    parser.add_argument("--remove_false_files", action="store_true",
                       help="실패 케이스 파일을 원본 디렉토리에서도 삭제")
    
    args = parser.parse_args()
    
    # 실패 케이스 디렉토리 생성
    os.makedirs(args.false_dir, exist_ok=True)
    
    # InsightFace 초기화
    print("InsightFace 모델 로딩 중...")
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='./models')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace 모델 로딩 완료!")
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 입력 디렉토리의 이미지 파일 목록
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    input_files = {f.stem: f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions}
    output_files = {f.stem: f for f in output_path.iterdir() 
                    if f.suffix.lower() in image_extensions}
    
    # 공통 파일명 찾기
    common_stems = set(input_files.keys()) & set(output_files.keys())
    
    print(f"\n입력 디렉토리: {len(input_files)}개 파일")
    print(f"출력 디렉토리: {len(output_files)}개 파일")
    print(f"공통 파일명: {len(common_stems)}개\n")
    
    # 통계
    success_count = 0
    fail_count = 0
    
    for idx, stem in enumerate(sorted(common_stems), 1):
        print(f"[{idx}/{len(common_stems)}] {stem}")
        
        input_file = input_files[stem]
        output_file = output_files[stem]
        
        try:
            is_valid, message = validate_image_pair(
                input_file, output_file, face_app,
                args.face_similarity_threshold,
                args.face_keypoint_threshold,
                args.pose_threshold
            )
            
            if is_valid:
                print(f"  ✓ 성공: {message}")
                success_count += 1
            else:
                print(f"  ✗ 실패: {message}")
                fail_count += 1
                
                # 실패 케이스 저장
                input_false_path = os.path.join(args.false_dir, f"{stem}_input{input_file.suffix}")
                output_false_path = os.path.join(args.false_dir, f"{stem}_output{output_file.suffix}")
                
                shutil.copy2(input_file, input_false_path)
                shutil.copy2(output_file, output_false_path)
                
                print(f"  → 실패 케이스 저장: {input_false_path}, {output_false_path}")
                
                # 원본 디렉토리에서 삭제 옵션
                if args.remove_false_files:
                    os.remove(input_file)
                    os.remove(output_file)
                    print(f"  → 원본 파일 삭제 완료")
        
        except Exception as e:
            print(f"  ✗ 오류 발생: {str(e)}")
            fail_count += 1
            
            # 오류 케이스도 실패로 저장
            try:
                input_false_path = os.path.join(args.false_dir, f"{stem}_input{input_file.suffix}")
                output_false_path = os.path.join(args.false_dir, f"{stem}_output{output_file.suffix}")
                
                shutil.copy2(input_file, input_false_path)
                shutil.copy2(output_file, output_false_path)
                
                # 원본 디렉토리에서 삭제 옵션
                if args.remove_false_files:
                    os.remove(input_file)
                    os.remove(output_file)
                    print(f"  → 원본 파일 삭제 완료")
            except:
                pass
    
    # 최종 통계
    print("\n" + "="*60)
    print(f"검증 완료!")
    print(f"총 처리: {len(common_stems)}개")
    print(f"성공: {success_count}개 ({success_count/len(common_stems)*100:.1f}%)")
    print(f"실패: {fail_count}개 ({fail_count/len(common_stems)*100:.1f}%)")
    print(f"실패 케이스 저장 경로: {args.false_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


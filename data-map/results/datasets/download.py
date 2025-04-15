from datasets import load_dataset
import os
# JSON 파일로 저장 (나중에 쉽게 불러오기 위해)
import json

# MMInstruction/VLFeedback 데이터셋 로드
dataset = load_dataset("MMInstruction/VLFeedback")

# 저장 디렉토리 설정
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# 이미지 저장
# 이미지 id가 중복되는지 확인
image_ids = set()
for i, example in enumerate(dataset['train']):
    image = example['image']  # PIL 이미지 객체
    image_id = example['id']

    file_path = os.path.join(output_dir, f"{image_id}.jpg")
    image.save(file_path)

    if i % 100 == 0:  # 진행 상황 출력
        print(f"{i}/{len(dataset['train'])} images saved.")

# 이미지-텍스트 매핑 저장
metadata = []
for i, example in enumerate(dataset['train']):
    image_id = example['id']
    prompt = example['prompt']  # 텍스트 프롬프트
    models = example['models']
    completions = example['completions']
    image_path = os.path.join(output_dir, f"{image_id}.jpg")
    metadata.append({"id": image_id, "prompt": prompt, "models": models, "completions": completions, "image_path": image_path})


with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("메타데이터 저장 완료!")
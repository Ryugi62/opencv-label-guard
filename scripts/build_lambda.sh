#!/bin/sh
# Lambda(python3.12, x86_64) 배포 꾸러미 3개를 dist/에 만든다: 계층 2개(cv2 / 공유 라이브러리+numpy) + 함수 zip(코드+가중치).
# 각 zip을 50MB(직접 업로드 한도) 미만으로 나눠 S3 없이 올린다. 풀면 합계 250MB 미만이어야 한다.
set -e
cd "$(dirname "$0")/.."
rm -rf dist build_lambda && mkdir -p dist build_lambda/site
${PIP:-.venv/bin/pip} install --quiet --target build_lambda/site --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --only-binary=:all: \
  -r requirements-lambda.txt
mkdir -p build_lambda/l1/python build_lambda/l2/python build_lambda/fn
mv build_lambda/site/cv2 build_lambda/l1/python/
mv build_lambda/site/opencv_python_headless.libs build_lambda/l2/python/
mv build_lambda/site/numpy build_lambda/site/numpy.libs build_lambda/l2/python/ 2>/dev/null || mv build_lambda/site/numpy build_lambda/l2/python/
cp -r label_guard build_lambda/fn/ && cp -r models build_lambda/fn/
find build_lambda -name __pycache__ -prune -exec rm -rf {} +
rm -f build_lambda/fn/label_guard/adapters/ocr_vision.swift
(cd build_lambda/l1 && zip -qr9 ../../dist/layer-cv2.zip python)
(cd build_lambda/l2 && zip -qr9 ../../dist/layer-libs-numpy.zip python)
(cd build_lambda/fn && zip -qr9 ../../dist/function.zip label_guard models)
ls -la dist; du -sm build_lambda/l1 build_lambda/l2 build_lambda/fn

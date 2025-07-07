#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
유틸리티 함수들
Hippocampal Signal Processing Project
"""

import math
import random
import time
from typing import List, Tuple, Optional, Dict


def fibonacci(n: int) -> List[int]:
    """피보나치 수열을 생성하는 함수"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence


def is_prime(n: int) -> bool:
    """소수 판별 함수"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def find_primes(limit: int) -> List[int]:
    """주어진 범위 내의 모든 소수를 찾는 함수"""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes


def bubble_sort(arr: List[int]) -> List[int]:
    """버블 정렬 알고리즘"""
    n = len(arr)
    arr_copy = arr.copy()
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
    
    return arr_copy


def binary_search(arr: List[int], target: int) -> Optional[int]:
    """이진 탐색 알고리즘"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None


def generate_random_data(size: int, min_val: int = 1, max_val: int = 100) -> List[int]:
    """랜덤 데이터 생성 함수"""
    return [random.randint(min_val, max_val) for _ in range(size)]


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """기본 통계 계산 함수"""
    if not data:
        return {}
    
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    
    return {
        "count": n,
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "min": min(data),
        "max": max(data)
    }


def timer_decorator(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper


@timer_decorator
def slow_function():
    """느린 함수 예제"""
    time.sleep(1)
    return "완료!"


def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """행렬 곱셈 함수"""
    if len(a[0]) != len(b):
        raise ValueError("행렬 차원이 맞지 않습니다.")
    
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    
    return result


def print_matrix(matrix: List[List[int]]):
    """행렬 출력 함수"""
    for row in matrix:
        print(row)


# 테스트 함수
def test_utils():
    """유틸리티 함수들 테스트"""
    print("=== 유틸리티 함수 테스트 ===")
    
    # 피보나치 수열
    print(f"피보나치 수열 (10개): {fibonacci(10)}")
    
    # 소수 찾기
    primes = find_primes(50)
    print(f"50 이하의 소수들: {primes}")
    
    # 랜덤 데이터 생성 및 정렬
    random_data = generate_random_data(10, 1, 100)
    print(f"랜덤 데이터: {random_data}")
    sorted_data = bubble_sort(random_data)
    print(f"정렬된 데이터: {sorted_data}")
    
    # 이진 탐색
    target = random_data[0]
    index = binary_search(sorted_data, target)
    print(f"숫자 {target}의 인덱스: {index}")
    
    # 통계 계산
    stats = calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"통계 정보: {stats}")
    
    # 행렬 곱셈
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[2, 0], [1, 2]]
    result_matrix = matrix_multiply(matrix_a, matrix_b)
    print("행렬 A:")
    print_matrix(matrix_a)
    print("행렬 B:")
    print_matrix(matrix_b)
    print("곱셈 결과:")
    print_matrix(result_matrix)
    
    # 데코레이터 테스트
    slow_function()


if __name__ == "__main__":
    test_utils() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Python 기본 코드 예제
Hippocampal Signal Processing Project
"""

import os
import sys
import datetime
import json
from typing import List, Dict, Any


def hello_world():
    """간단한 인사말 함수"""
    print("안녕하세요! Python 기본 코드에 오신 것을 환영합니다.")
    print("=" * 50)


def basic_calculator():
    """기본 계산기 함수"""
    print("\n=== 기본 계산기 ===")
    
    try:
        num1 = float(input("첫 번째 숫자를 입력하세요: "))
        num2 = float(input("두 번째 숫자를 입력하세요: "))
        operation = input("연산을 선택하세요 (+, -, *, /): ")
        
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            if num2 == 0:
                print("오류: 0으로 나눌 수 없습니다.")
                return
            result = num1 / num2
        else:
            print("잘못된 연산자입니다.")
            return
            
        print(f"결과: {num1} {operation} {num2} = {result}")
        
    except ValueError:
        print("올바른 숫자를 입력해주세요.")


def list_operations():
    """리스트 조작 예제"""
    print("\n=== 리스트 조작 예제 ===")
    
    # 기본 리스트 생성
    numbers = [1, 2, 3, 4, 5]
    print(f"원본 리스트: {numbers}")
    
    # 리스트에 요소 추가
    numbers.append(6)
    print(f"append(6) 후: {numbers}")
    
    # 리스트 정렬
    numbers.sort(reverse=True)
    print(f"내림차순 정렬: {numbers}")
    
    # 리스트 컴프리헨션
    squares = [x**2 for x in numbers]
    print(f"제곱 리스트: {squares}")
    
    # 필터링
    even_numbers = [x for x in numbers if x % 2 == 0]
    print(f"짝수만: {even_numbers}")


def dictionary_example():
    """딕셔너리 예제"""
    print("\n=== 딕셔너리 예제 ===")
    
    # 학생 정보 딕셔너리
    student = {
        "name": "홍길동",
        "age": 20,
        "major": "컴퓨터공학",
        "grades": [85, 90, 78, 92, 88]
    }
    
    print(f"학생 정보: {student}")
    print(f"이름: {student['name']}")
    print(f"평균 점수: {sum(student['grades']) / len(student['grades']):.2f}")
    
    # 딕셔너리 업데이트
    student["email"] = "hong@example.com"
    print(f"이메일 추가 후: {student}")


def file_operations():
    """파일 조작 예제"""
    print("\n=== 파일 조작 예제 ===")
    
    # 파일 쓰기
    filename = "sample.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("안녕하세요!\n")
            f.write("이것은 Python으로 작성된 파일입니다.\n")
            f.write(f"작성 시간: {datetime.datetime.now()}\n")
        print(f"파일 '{filename}'이 생성되었습니다.")
        
        # 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"파일 내용:\n{content}")
            
    except Exception as e:
        print(f"파일 조작 중 오류 발생: {e}")


def class_example():
    """클래스 예제"""
    print("\n=== 클래스 예제 ===")
    
    class Person:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age
            
        def introduce(self):
            return f"안녕하세요! 저는 {self.name}이고, {self.age}살입니다."
            
        def birthday(self):
            self.age += 1
            return f"{self.name}의 생일! 나이가 {self.age}살이 되었습니다."
    
    # Person 객체 생성
    person1 = Person("김철수", 25)
    person2 = Person("이영희", 23)
    
    print(person1.introduce())
    print(person2.introduce())
    print(person1.birthday())


def main():
    """메인 함수"""
    hello_world()
    
    # 각 예제 함수 실행
    basic_calculator()
    list_operations()
    dictionary_example()
    file_operations()
    class_example()
    
    print("\n" + "=" * 50)
    print("모든 예제가 완료되었습니다!")
    print("Python 기본 문법을 연습해보세요.")


if __name__ == "__main__":
    main() 
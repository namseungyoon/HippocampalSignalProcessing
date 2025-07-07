#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 테스트 케이스
Hippocampal Signal Processing Project
"""

import unittest
from utils import fibonacci, is_prime, bubble_sort, binary_search, calculate_statistics


class TestUtils(unittest.TestCase):
    """유틸리티 함수들에 대한 테스트 클래스"""
    
    def test_fibonacci(self):
        """피보나치 수열 테스트"""
        # 기본 케이스
        self.assertEqual(fibonacci(0), [])
        self.assertEqual(fibonacci(1), [0])
        self.assertEqual(fibonacci(2), [0, 1])
        self.assertEqual(fibonacci(5), [0, 1, 1, 2, 3])
        
        # 음수 케이스
        self.assertEqual(fibonacci(-1), [])
    
    def test_is_prime(self):
        """소수 판별 테스트"""
        # 소수들
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))
        self.assertTrue(is_prime(11))
        
        # 합성수들
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))
        self.assertFalse(is_prime(8))
        self.assertFalse(is_prime(9))
    
    def test_bubble_sort(self):
        """버블 정렬 테스트"""
        # 일반적인 케이스
        test_list = [64, 34, 25, 12, 22, 11, 90]
        sorted_list = bubble_sort(test_list)
        self.assertEqual(sorted_list, [11, 12, 22, 25, 34, 64, 90])
        
        # 빈 리스트
        self.assertEqual(bubble_sort([]), [])
        
        # 이미 정렬된 리스트
        sorted_input = [1, 2, 3, 4, 5]
        self.assertEqual(bubble_sort(sorted_input), [1, 2, 3, 4, 5])
        
        # 역순 정렬된 리스트
        reverse_input = [5, 4, 3, 2, 1]
        self.assertEqual(bubble_sort(reverse_input), [1, 2, 3, 4, 5])
    
    def test_binary_search(self):
        """이진 탐색 테스트"""
        # 정렬된 리스트
        test_list = [1, 3, 5, 7, 9, 11, 13, 15]
        
        # 존재하는 요소들
        self.assertEqual(binary_search(test_list, 1), 0)
        self.assertEqual(binary_search(test_list, 7), 3)
        self.assertEqual(binary_search(test_list, 15), 7)
        
        # 존재하지 않는 요소들
        self.assertIsNone(binary_search(test_list, 2))
        self.assertIsNone(binary_search(test_list, 10))
        self.assertIsNone(binary_search(test_list, 20))
        
        # 빈 리스트
        self.assertIsNone(binary_search([], 5))
    
    def test_calculate_statistics(self):
        """통계 계산 테스트"""
        # 일반적인 케이스
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_statistics(data)
        
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        
        # 빈 리스트
        empty_stats = calculate_statistics([])
        self.assertEqual(empty_stats, {})
        
        # 단일 요소
        single_stats = calculate_statistics([42.0])
        self.assertEqual(single_stats["mean"], 42.0)
        self.assertEqual(single_stats["variance"], 0.0)


class TestMainFunctions(unittest.TestCase):
    """메인 함수들에 대한 테스트 클래스"""
    
    def test_hello_world(self):
        """hello_world 함수 테스트"""
        # 이 함수는 단순히 출력만 하므로 예외가 발생하지 않는지만 확인
        try:
            from main import hello_world
            hello_world()
            # 함수가 정상적으로 실행되면 테스트 통과
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"hello_world 함수 실행 중 오류 발생: {e}")
    
    def test_list_operations(self):
        """list_operations 함수 테스트"""
        try:
            from main import list_operations
            list_operations()
            # 함수가 정상적으로 실행되면 테스트 통과
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"list_operations 함수 실행 중 오류 발생: {e}")
    
    def test_dictionary_example(self):
        """dictionary_example 함수 테스트"""
        try:
            from main import dictionary_example
            dictionary_example()
            # 함수가 정상적으로 실행되면 테스트 통과
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"dictionary_example 함수 실행 중 오류 발생: {e}")


def run_tests():
    """모든 테스트를 실행하는 함수"""
    print("=== 테스트 시작 ===")
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 클래스들 추가
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestMainFunctions))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 출력
    print(f"\n=== 테스트 결과 ===")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"실패한 테스트: {len(result.failures)}")
    print(f"오류 발생: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n오류 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 테스트 실행
    success = run_tests()
    
    if success:
        print("\n🎉 모든 테스트가 통과했습니다!")
    else:
        print("\n❌ 일부 테스트가 실패했습니다.") 
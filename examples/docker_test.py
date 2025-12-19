#!/usr/bin/env python3
"""
Docker test script for rust-jieba
"""

import rust_jieba
import jieba

def test_basic_segmentation():
    """Test basic segmentation functionality"""
    test_text = "我爱北京天安门"

    print("=== Rust jieba Test ===")
    rust_result = list(rust_jieba.cut(test_text))
    print(f"Input: {test_text}")
    print(f"Rust jieba: {'/'.join(rust_result)}")

    print("\n=== Python jieba Test ===")
    py_result = list(jieba.cut(test_text))
    print(f"Python jieba: {'/'.join(py_result)}")

    print(f"\nResults match: {rust_result == py_result}")

    # Test different modes
    print("\n=== Different Modes ===")

    # Full mode
    rust_full = list(rust_jieba.cut(test_text, cut_all=True))
    py_full = list(jieba.cut(test_text, cut_all=True))
    print(f"Full mode - Rust: {'/'.join(rust_full)}")
    print(f"Full mode - Python: {'/'.join(py_full)}")

    # Search mode
    search_text = "小明硕士毕业于中国科学院计算所"
    rust_search = list(rust_jieba.cut_for_search(search_text))
    py_search = list(jieba.cut_for_search(search_text))
    print(f"\nSearch mode - Rust: {'/'.join(rust_search)}")
    print(f"Search mode - Python: {'/'.join(py_search)}")

    # Tokenize
    print(f"\n=== Tokenize Test ===")
    tokens = rust_jieba.tokenize(test_text)
    print("Tokens:")
    for word, flag, start, end in tokens:
        print(f"  {word} ({flag}): [{start}:{end})")

def test_performance():
    """Test performance with larger text"""
    import time

    text = "在这个快速发展的时代，人工智能技术正在改变我们的生活方式。" * 100

    print(f"\n=== Performance Test ===")
    print(f"Text length: {len(text)} characters")

    # Test Rust jieba
    start_time = time.time()
    for _ in range(100):
        _ = list(rust_jieba.cut(text))
    rust_time = time.time() - start_time

    # Test Python jieba
    start_time = time.time()
    for _ in range(100):
        _ = list(jieba.cut(text))
    py_time = time.time() - start_time

    print(f"Rust jieba: {rust_time:.4f}s for 100 iterations")
    print(f"Python jieba: {py_time:.4f}s for 100 iterations")
    print(f"Speedup: {py_time/rust_time:.2f}x")

if __name__ == "__main__":
    print("Rust jieba Docker Test")
    print("=" * 40)

    try:
        test_basic_segmentation()
        test_performance()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
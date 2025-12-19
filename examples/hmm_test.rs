use rust_jieba::Jieba;

fn main() {
    let jieba = Jieba::new();

    // 测试用例：包含一些可能不在词典中的字符组合
    let test_cases = vec![
        ("北京大学的计算机系学生", "测试词典词和HMM"),
        ("我是研究生", "测试包含可能的未登录词"),
        ("自然语言处理技术", "测试复合词"),
        ("我爱编程", "测试简单句子"),
    ];

    println!("=== HMM分词测试 ===");
    for (text, desc) in test_cases {
        println!("\n{}: {}", desc, text);

        // 不使用HMM
        let result_no_hmm = jieba.cut(text, false);
        println!("不使用HMM: {:?}", result_no_hmm);

        // 使用HMM
        let result_with_hmm = jieba.cut(text, true);
        println!("使用HMM:   {:?}", result_with_hmm);

        // 比较差异
        if result_no_hmm != result_with_hmm {
            println!("⭐ HMM产生了不同的结果!");
        } else {
            println!("➡️  两种方法结果相同");
        }
    }
}
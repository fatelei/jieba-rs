use rust_jieba::Jieba;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
struct PerformanceStats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Clone)]
struct AccuracyResult {
    text: String,
    expected: Vec<String>,
    no_hmm: Vec<String>,
    hmm: Vec<String>,
    search: Vec<String>,
}

fn test_rust_jieba_performance(
    jieba: &Jieba,
    test_sentences: &[String],
    iterations: usize,
) -> HashMap<String, PerformanceStats> {
    println!("=== Rust jieba 性能测试 ===");

    let mut results = HashMap::new();

    // 预热
    for sentence in test_sentences.iter().take(5) {
        jieba.cut(sentence, false);
        jieba.cut(sentence, true);
    }

    // 测试不使用HMM
    let mut times_no_hmm = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        for sentence in test_sentences {
            jieba.cut(sentence, false);
        }
        let duration = start.elapsed().as_secs_f64();
        times_no_hmm.push(duration);
    }

    // 测试使用HMM
    let mut times_hmm = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        for sentence in test_sentences {
            jieba.cut(sentence, true);
        }
        let duration = start.elapsed().as_secs_f64();
        times_hmm.push(duration);
    }

    // 测试全模式
    let mut times_full = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        for sentence in test_sentences {
            jieba.cut_full(sentence, true);
        }
        let duration = start.elapsed().as_secs_f64();
        times_full.push(duration);
    }

    // 计算统计信息
    results.insert("no_hmm".to_string(), calculate_stats(&times_no_hmm));
    results.insert("hmm".to_string(), calculate_stats(&times_hmm));
    results.insert("full".to_string(), calculate_stats(&times_full));

    results
}

fn calculate_stats(times: &[f64]) -> PerformanceStats {
    let mut sorted_times = times.to_vec();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = if sorted_times.len() % 2 == 0 {
        (sorted_times[sorted_times.len() / 2 - 1] + sorted_times[sorted_times.len() / 2]) / 2.0
    } else {
        sorted_times[sorted_times.len() / 2]
    };

    PerformanceStats {
        mean,
        median,
        min: sorted_times[0],
        max: sorted_times[sorted_times.len() - 1],
    }
}

fn test_rust_jieba_accuracy(
    jieba: &Jieba,
    test_cases: &[(&str, Vec<String>)],
) -> Vec<AccuracyResult> {
    println!("=== Rust jieba 准确率测试 ===");

    let mut results = Vec::new();

    for (text, expected) in test_cases {
        let no_hmm = jieba.cut(text, false);
        let hmm = jieba.cut(text, true);
        let search = jieba.cut_for_search(text, true);

        let result = AccuracyResult {
            text: text.to_string(),
            expected: expected.clone(),
            no_hmm,
            hmm,
            search,
        };

        println!("文本: {}", text);
        println!("期望: {:?}", expected);
        println!("精确模式(HMM=false): {:?}", result.no_hmm);
        println!("精确模式(HMM=true):  {:?}", result.hmm);
        println!("搜索引擎模式:         {:?}", result.search);
        println!();

        results.push(result);
    }

    results
}

fn calculate_accuracy(
    python_results: &[AccuracyResult],
    rust_results: &[AccuracyResult],
) -> (f64, f64, f64) {
    assert_eq!(python_results.len(), rust_results.len());

    let mut no_hmm_matches = 0;
    let mut hmm_matches = 0;
    let mut search_matches = 0;

    for (py_res, rust_res) in python_results.iter().zip(rust_results.iter()) {
        // 计算与期望结果的匹配度
        let expected_no_hmm = &py_res.expected;
        let expected_hmm = &py_res.expected;

        // 简单的完全匹配比较
        if rust_res.no_hmm == *expected_no_hmm {
            no_hmm_matches += 1;
        }

        if rust_res.hmm == *expected_hmm {
            hmm_matches += 1;
        }

        if rust_res.search == py_res.search {
            search_matches += 1;
        }
    }

    let total = python_results.len() as f64;
    (
        no_hmm_matches as f64 / total * 100.0,
        hmm_matches as f64 / total * 100.0,
        search_matches as f64 / total * 100.0,
    )
}

fn main() {
    let jieba = Jieba::new();

    // 测试数据
    let test_sentences = vec![
        "北京大学的计算机系学生正在研究自然语言处理和机器学习算法。".to_string(),
        "我是一个学生，热爱编程和人工智能技术。".to_string(),
        "自然语言处理是人工智能的重要分支领域。".to_string(),
        "我爱北京天安门，天安门上太阳升。".to_string(),
        "上海浦东开发区是中国改革开放的重要窗口。".to_string(),
        "机器学习和深度学习技术正在快速发展。".to_string(),
        "这个算法的时间复杂度是O(n log n)。".to_string(),
        "人工智能将在未来改变我们的生活方式。".to_string(),
        "中文分词技术对于自然语言处理至关重要。".to_string(),
        "深度学习模型需要大量的训练数据支持。".to_string(),
        "科学研究需要严谨的态度和创新的精神。".to_string(),
        "技术的进步离不开基础理论的支撑。".to_string(),
        "大数据时代背景下，数据挖掘技术变得越来越重要。".to_string(),
        "云计算和边缘计算是分布式计算的两种重要形式。".to_string(),
        "量子计算有潜力解决传统计算难以处理的复杂问题。".to_string(),
    ];

    let accuracy_test_cases: Vec<(&str, Vec<String>)> = vec![
        // 基于Python jieba标准结果更新
        (
            "北京大学的计算机系学生",
            vec![
                "北京大学".to_string(),
                "的".to_string(),
                "计算机系".to_string(),
                "学生".to_string(),
            ],
        ),
        (
            "我是一个学生",
            vec![
                "我".to_string(),
                "是".to_string(),
                "一个".to_string(),
                "学生".to_string(),
            ],
        ),
        (
            "我爱北京天安门",
            vec![
                "我".to_string(),
                "爱".to_string(),
                "北京".to_string(),
                "天安门".to_string(),
            ],
        ),
        (
            "自然语言处理是人工智能",
            vec![
                "自然语言".to_string(),
                "处理".to_string(),
                "是".to_string(),
                "人工智能".to_string(),
            ],
        ),
        (
            "研究生命科学",
            vec!["研究".to_string(), "生命科学".to_string()],
        ),
        (
            "机器学习算法",
            vec!["机器".to_string(), "学习".to_string(), "算法".to_string()],
        ),
        (
            "深度学习模型",
            vec!["深度".to_string(), "学习".to_string(), "模型".to_string()],
        ),
        (
            "计算机科学技术",
            vec!["计算机".to_string(), "科学技术".to_string()],
        ),
        (
            "人工智能研究",
            vec!["人工智能".to_string(), "研究".to_string()],
        ),
        (
            "数据挖掘技术",
            vec!["数据挖掘".to_string(), "技术".to_string()],
        ),
        // 技术相关 - 基于Python标准结果
        (
            "云计算和大数据技术",
            vec![
                "云".to_string(),
                "计算".to_string(),
                "和".to_string(),
                "大".to_string(),
                "数据".to_string(),
                "技术".to_string(),
            ],
        ),
        (
            "区块链技术在金融领域的应用",
            vec![
                "区块".to_string(),
                "链".to_string(),
                "技术".to_string(),
                "在".to_string(),
                "金融".to_string(),
                "领域".to_string(),
                "的".to_string(),
                "应用".to_string(),
            ],
        ),
        (
            "物联网设备的智能控制系统",
            vec![
                "物".to_string(),
                "联网".to_string(),
                "设备".to_string(),
                "的".to_string(),
                "智能".to_string(),
                "控制系统".to_string(),
            ],
        ),
        (
            "5G网络通信技术标准",
            vec![
                "5G".to_string(),
                "网络通信".to_string(),
                "技术标准".to_string(),
            ],
        ),
        // 生活相关 - 基于Python标准结果
        (
            "今天天气真不错",
            vec!["今天天气".to_string(), "真不错".to_string()],
        ),
        (
            "这家餐厅的菜很好吃",
            vec![
                "这家".to_string(),
                "餐厅".to_string(),
                "的".to_string(),
                "菜".to_string(),
                "很".to_string(),
                "好吃".to_string(),
            ],
        ),
        (
            "我喜欢听音乐和看电影",
            vec![
                "我".to_string(),
                "喜欢".to_string(),
                "听".to_string(),
                "音乐".to_string(),
                "和".to_string(),
                "看".to_string(),
                "电影".to_string(),
            ],
        ),
        (
            "周末我们去公园散步吧",
            vec![
                "周末".to_string(),
                "我们".to_string(),
                "去".to_string(),
                "公园".to_string(),
                "散步".to_string(),
                "吧".to_string(),
            ],
        ),
        // 新闻相关 - 基于Python标准结果
        (
            "国家主席发表重要讲话",
            vec![
                "国家".to_string(),
                "主席".to_string(),
                "发表".to_string(),
                "重要讲话".to_string(),
            ],
        ),
        (
            "经济全球化发展趋势",
            vec![
                "经济".to_string(),
                "全球化".to_string(),
                "发展趋势".to_string(),
            ],
        ),
        (
            "环境保护问题引起广泛关注",
            vec![
                "环境保护".to_string(),
                "问题".to_string(),
                "引起".to_string(),
                "广泛".to_string(),
                "关注".to_string(),
            ],
        ),
        (
            "科技创新推动产业升级",
            vec![
                "科技".to_string(),
                "创新".to_string(),
                "推动".to_string(),
                "产业".to_string(),
                "升级".to_string(),
            ],
        ),
        // 文学相关 - 基于Python标准结果
        (
            "诗词歌赋是中国传统文化",
            vec![
                "诗词歌赋".to_string(),
                "是".to_string(),
                "中国".to_string(),
                "传统".to_string(),
                "文化".to_string(),
            ],
        ),
        (
            "红楼梦是中国古典文学名著",
            vec![
                "红楼梦".to_string(),
                "是".to_string(),
                "中国".to_string(),
                "古典文学".to_string(),
                "名著".to_string(),
            ],
        ),
        (
            "鲁迅是中国现代文学的重要作家",
            vec![
                "鲁迅".to_string(),
                "是".to_string(),
                "中国".to_string(),
                "现代文学".to_string(),
                "的".to_string(),
                "重要".to_string(),
                "作家".to_string(),
            ],
        ),
        // 数字和特殊字符混合 - 基于Python标准结果
        (
            "2023年中国GDP增长5.2%",
            vec![
                "2023".to_string(),
                "年".to_string(),
                "中国".to_string(),
                "GDP".to_string(),
                "增长".to_string(),
                "5.2%".to_string(),
            ],
        ),
        (
            "Python3.8版本发布",
            vec![
                "Python3.8".to_string(),
                "版本".to_string(),
                "发布".to_string(),
            ],
        ),
        (
            "手机号码是13800138000",
            vec![
                "手机号码".to_string(),
                "是".to_string(),
                "13800138000".to_string(),
            ],
        ),
        // 长句 - 基于Python标准结果
        (
            "在这个快速发展的时代，我们需要不断学习新知识来适应社会的变化",
            vec![
                "在".to_string(),
                "这个".to_string(),
                "快速".to_string(),
                "发展".to_string(),
                "的".to_string(),
                "时代".to_string(),
                "，".to_string(),
                "我们".to_string(),
                "需要".to_string(),
                "不断".to_string(),
                "学习".to_string(),
                "新".to_string(),
                "知识".to_string(),
                "来".to_string(),
                "适应".to_string(),
                "社会".to_string(),
                "的".to_string(),
                "变化".to_string(),
            ],
        ),
    ];

    println!("测试句子数量: {}", test_sentences.len());
    println!("准确率测试用例: {}", accuracy_test_cases.len());
    println!();

    // 性能测试
    let perf_results = test_rust_jieba_performance(&jieba, &test_sentences, 50);

    println!("\n=== Rust jieba 性能结果 (单位: 秒) ===");
    for (mode, stats) in &perf_results {
        println!("{} 模式:", mode.to_uppercase());
        println!("  平均时间: {:.6}s", stats.mean);
        println!("  中位数时间: {:.6}s", stats.median);
        println!("  最短时间: {:.6}s", stats.min);
        println!("  最长时间: {:.6}s", stats.max);
        println!();
    }

    // 准确率测试
    let accuracy_results = test_rust_jieba_accuracy(&jieba, &accuracy_test_cases);

    // 检查HMM是否真的产生了不同的结果
    let hmm_diff_count = accuracy_results
        .iter()
        .filter(|r| r.no_hmm != r.hmm)
        .count();

    println!(
        "HMM产生不同结果的用例数: {}/{}",
        hmm_diff_count,
        accuracy_results.len()
    );

    if hmm_diff_count > 0 {
        println!("✅ HMM功能正常工作");
    } else {
        println!("⚠️  HMM可能没有产生预期效果");
    }
}

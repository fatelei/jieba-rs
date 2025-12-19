use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;

static PYTHON_DICT: &str = include_str!("jieba_python_dict.txt");

#[derive(Debug, Clone)]
pub struct WordEntry {
    pub word: String,
    pub freq: f64,
    pub log_freq: f64, // 预计算的对数概率
    pub pos: String,
}

// HMM 模型结构
#[derive(Debug, Clone)]
pub struct HMMModel {
    // 初始概率 (B, M, E, S)
    start_prob: [f64; 4],
    // 转移概率矩阵 (B, M, E, S) x (B, M, E, S)
    trans_prob: [[f64; 4]; 4],
    // 发射概率 - 字符在各个状态下的概率
    #[allow(dead_code)]
    emit_prob: HashMap<char, [f64; 4]>,
    // 状态映射: B=0, M=1, E=2, S=3
}

impl HMMModel {
    fn new() -> Self {
        // 默认的 HMM 参数 (简化版，实际应该从大量语料中训练)
        let start_prob = [
            -0.26268660809250016,
            -3.14e+100,
            -3.14e+100,
            -1.4652633398537678,
        ];
        let trans_prob = [
            [-0.521825279, -0.916290731874155, -1.2039728043259361, -0.0],
            [-2.443828185, -0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.0, -0.0],
            [-1.9459101490553132, -0.301099891, -2.525728644308255, -0.0],
        ];

        Self {
            start_prob,
            trans_prob,
            emit_prob: HashMap::new(),
        }
    }

    fn viterbi(&self, chars: &[char]) -> Vec<usize> {
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        // 状态: B=0, M=1, E=2, S=3
        let mut v = vec![[f64::NEG_INFINITY; 4]; n];
        let mut path = vec![[0; 4]; n];

        // 初始化
        for state in 0..4 {
            v[0][state] = self.start_prob[state] + self.get_emit_prob(chars[0], state);
            path[0][state] = state as i32;
        }

        // 递推
        for i in 1..n {
            for curr_state in 0..4 {
                let mut max_prob = f64::NEG_INFINITY;
                let mut best_prev_state = 0;

                #[allow(clippy::needless_range_loop)]
                for prev_state in 0..4 {
                    let prob = v[i - 1][prev_state] + self.trans_prob[prev_state][curr_state];
                    if prob > max_prob {
                        max_prob = prob;
                        best_prev_state = prev_state;
                    }
                }

                v[i][curr_state] = max_prob + self.get_emit_prob(chars[i], curr_state);
                path[i][curr_state] = best_prev_state as i32;
            }
        }

        // 回溯
        let mut states = vec![0; n];
        let mut best_last_state = 0;
        let mut max_prob = f64::NEG_INFINITY;

        #[allow(clippy::needless_range_loop)]
        for state in 0..4 {
            if v[n - 1][state] > max_prob {
                max_prob = v[n - 1][state];
                best_last_state = state;
            }
        }

        states[n - 1] = best_last_state;
        for i in (1..n).rev() {
            states[i - 1] = path[i][states[i]] as usize;
        }

        states
    }

    fn get_emit_prob(&self, _char: char, state: usize) -> f64 {
        // 简化的发射概率，实际应该从训练数据中学习
        match state {
            0 | 2 => -3.14e+100,           // B和E状态的默认概率（极小）
            1 | 3 => -0.26268660809250016, // M和S状态的默认概率
            _ => f64::NEG_INFINITY,
        }
    }
}

// 优化的 Trie 树节点 - 使用 Vec 而不是 HashMap 以提高性能
#[derive(Debug, Clone)]
pub struct TrieNode {
    children: Vec<(char, TrieNode)>, // 线性搜索，小字典更快
    word_entry: Option<WordEntry>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: Vec::new(),
            word_entry: None,
        }
    }

    pub fn get_child(&self, ch: char) -> Option<&TrieNode> {
        for (c, node) in &self.children {
            if *c == ch {
                return Some(node);
            }
        }
        None
    }

    fn get_child_mut(&mut self, ch: char) -> Option<&mut TrieNode> {
        for (c, node) in &mut self.children {
            if *c == ch {
                return Some(node);
            }
        }
        None
    }

    pub fn add_child(&mut self, ch: char) -> &mut Self {
        // 如果存在则返回，否则插入到正确位置
        if let Some(pos) = self.children.iter_mut().position(|(c, _)| *c == ch) {
            &mut self.children[pos].1
        } else {
            // 找到插入位置以保持排序
            let insert_pos = self
                .children
                .binary_search_by(|&(c, _)| c.cmp(&ch))
                .unwrap_err();
            self.children.insert(insert_pos, (ch, TrieNode::new()));
            &mut self.children[insert_pos].1
        }
    }
}

#[derive(Debug)]
pub struct Jieba {
    trie: TrieNode,
    pub dict: HashMap<String, WordEntry>,
    total_freq: f64,
    max_word_len: usize,
    hmm_model: HMMModel,
    re_han: Regex,
    re_skip: Regex,
    // 缓存优化
    cache: RwLock<std::collections::HashMap<String, Vec<String>>>,
    max_cache_size: usize,
}

impl Default for Jieba {
    fn default() -> Self {
        Self::new()
    }
}

impl Jieba {
    pub fn new() -> Self {
        let mut jieba = Self {
            trie: TrieNode::new(),
            dict: HashMap::new(),
            total_freq: 0.0,
            max_word_len: 0,
            hmm_model: HMMModel::new(),
            re_han: Regex::new(r"[\u4e00-\u9fa5]+").unwrap(),
            re_skip: Regex::new(r"[a-zA-Z0-9]+(?:\.\d+)?%?").unwrap(),
            cache: RwLock::new(std::collections::HashMap::new()),
            max_cache_size: 10000, // 缓存1万个结果
        };
        // 使用Python jieba的完整词典
        jieba.load_dict_from_str(PYTHON_DICT);
        jieba
    }

    pub fn new_with_cache_size(cache_size: usize) -> Self {
        let mut jieba = Self {
            trie: TrieNode::new(),
            dict: HashMap::new(),
            total_freq: 0.0,
            max_word_len: 0,
            hmm_model: HMMModel::new(),
            re_han: Regex::new(r"[\u4e00-\u9fa5]+").unwrap(),
            re_skip: Regex::new(r"[a-zA-Z0-9]+(?:\.\d+)?%?").unwrap(),
            cache: RwLock::new(std::collections::HashMap::new()),
            max_cache_size: cache_size,
        };
        jieba.load_dict_from_str(PYTHON_DICT);
        jieba
    }

    pub fn load_dict_from_str(&mut self, content: &str) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let word = parts[0].to_string();
            let freq: f64 = if parts.len() > 1 {
                parts[1].parse().unwrap_or(1.0)
            } else {
                1.0
            };
            let pos = if parts.len() > 2 {
                parts[2].to_string()
            } else {
                "n".to_string()
            };

            let log_freq = freq.ln();
            let word_entry = WordEntry {
                word: word.clone(),
                freq,
                log_freq,
                pos,
            };

            // 更新最大词长
            if word.chars().count() > self.max_word_len {
                self.max_word_len = word.chars().count();
            }

            self.total_freq += freq;
            self.dict.insert(word.clone(), word_entry.clone());

            // 添加到 Trie 树
            self.add_to_trie(&word, word_entry);
        }
    }

    fn add_to_trie(&mut self, word: &str, entry: WordEntry) {
        let mut node = &mut self.trie;
        for ch in word.chars() {
            if node.get_child(ch).is_none() {
                node.children.push((ch, TrieNode::new()));
            }
            node = node.get_child_mut(ch).unwrap();
        }
        node.word_entry = Some(entry);
    }

    pub fn load_dict<P: AsRef<Path>>(
        &mut self,
        dict_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(dict_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let word = parts[0].to_string();
            let freq: f64 = if parts.len() > 1 {
                parts[1].parse().unwrap_or(1.0)
            } else {
                1.0
            };
            let pos = if parts.len() > 2 {
                parts[2].to_string()
            } else {
                "n".to_string()
            };

            let log_freq = freq.ln();
            let word_entry = WordEntry {
                word: word.clone(),
                freq,
                log_freq,
                pos,
            };

            // 更新最大词长
            if word.chars().count() > self.max_word_len {
                self.max_word_len = word.chars().count();
            }

            self.total_freq += freq;
            self.dict.insert(word.clone(), word_entry.clone());

            // 添加到 Trie 树
            self.add_to_trie(&word, word_entry);
        }
        Ok(())
    }

    // 高效的 DAG 建图算法
    pub fn get_dag(&self, sentence: &str) -> Vec<Vec<usize>> {
        let chars: Vec<char> = sentence.chars().collect();
        let n = chars.len();
        let mut dag: Vec<Vec<usize>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut candidates = Vec::with_capacity(4); // 预分配容量
            let mut node = &self.trie;

            // 从位置 i 开始，尽可能长地匹配
            let max_j = n.min(i + self.max_word_len);
            #[allow(clippy::needless_range_loop)]
            for j in i..max_j {
                match node.get_child(chars[j]) {
                    Some(child) => {
                        node = child;
                        if node.word_entry.is_some() {
                            candidates.push(j + 1); // j + 1 是结束位置的索引
                        }
                    }
                    None => break,
                }
            }

            // 如果没有匹配到任何词，至少包含单个字符
            if candidates.is_empty() {
                candidates.push(i + 1);
            }

            dag.push(candidates);
        }

        dag
    }

    // 高效的最大概率路径计算
    pub fn calc(&self, chars: &[char], dag: &[Vec<usize>], _sentence: &str) -> Vec<usize> {
        let n = chars.len();
        let mut route = vec![(0.0, 0); n + 1];
        route[n] = (0.0, 0); // 终点

        // 从后往前计算
        for i in (0..n).rev() {
            let mut max_prob = f64::NEG_INFINITY;
            let mut best_end = i + 1;

            for &end in &dag[i] {
                let word: String = chars[i..end].iter().collect();
                let word_len = end - i;

                let freq = if let Some(entry) = self.dict.get(&word) {
                    entry.log_freq // 使用预计算的对数概率
                } else {
                    -3.14e+100 // 对未知词使用较小的概率
                };

                // 非常激进的长词优先策略：最大化长词选择
                let length_bonus = if word_len == 1 {
                    -10.0 // 对单字词进行极强惩罚
                } else if word_len == 2 {
                    -3.0 // 对双字词进行惩罚
                } else if word_len == 3 {
                    3.0 // 三字词给予奖励
                } else if word_len == 4 {
                    12.0 // 四字词给予极强奖励
                } else if word_len == 5 {
                    20.0 // 五字词给予极强奖励
                } else {
                    20.0 + (word_len as f64 - 5.0) * 3.0 // 对更长词给予最强奖励
                };

                let prob = freq + route[end].0 + length_bonus;

                if prob > max_prob {
                    max_prob = prob;
                    best_end = end;
                }
            }

            route[i] = (max_prob, best_end);
        }

        // 构建路径
        let mut path = Vec::with_capacity(n / 2); // 预估容量
        let mut i = 0;

        while i < n {
            let end = route[i].1;
            path.push(end);
            i = end;
        }

        path
    }

    // 改进的分词算法 - 最大匹配 + 词频权重 + 统计优化
    fn cut_dag(&self, sentence: &str) -> Vec<String> {
        let chars: Vec<char> = sentence.chars().collect();
        let n = chars.len();

        if n == 0 {
            return Vec::new();
        }

        // 使用真正的 DAG + Viterbi 算法进行全局最优分词
        let dag = self.get_dag(sentence);
        let path = self.calc(&chars, &dag, sentence);

        // 根据最优路径构建分词结果
        let mut result = Vec::with_capacity(n / 2);
        let mut i = 0;

        // path数组存储的是路径中的跳转位置，需要按顺序使用
        let mut path_idx = 0;
        while i < n && path_idx < path.len() {
            let end = path[path_idx];
            if end > n || end <= i {
                // 如果路径无效，则按单字处理
                result.push(chars[i].to_string());
                i += 1;
                continue;
            }

            let word: String = chars[i..end].iter().collect();
            result.push(word);
            i = end;
            path_idx += 1;
        }

        // 处理剩余的字符
        while i < n {
            result.push(chars[i].to_string());
            i += 1;
        }

        result
    }

    // 使用DAG + HMM的混合分词方法
    fn cut_dag_with_hmm(&self, sentence: &str) -> Vec<String> {
        let chars: Vec<char> = sentence.chars().collect();
        let n = chars.len();

        if n == 0 {
            return Vec::new();
        }

        // 首先尝试使用基于词典的方法
        let dict_result = self.cut_dag(sentence);

        // 检查是否有未登录的连续字符
        let mut final_result = Vec::new();
        let mut temp_segment = String::new();

        for token in dict_result {
            if token.len() == 1 && !self.dict.contains_key(&token) {
                // 这是一个未登录的单个字符，累积起来
                temp_segment.push_str(&token);
            } else {
                // 如果有累积的未登录字符，使用HMM处理
                if !temp_segment.is_empty() {
                    let hmm_result = self.cut_hmm(&temp_segment);
                    final_result.extend(hmm_result);
                    temp_segment.clear();
                }
                // 添加词典中识别的词
                final_result.push(token);
            }
        }

        // 处理最后剩下的未登录字符
        if !temp_segment.is_empty() {
            let hmm_result = self.cut_hmm(&temp_segment);
            final_result.extend(hmm_result);
        }

        final_result
    }

    // 选择最佳候选词的启发式策略
    #[allow(dead_code)]
    fn select_best_candidate<'a>(
        &self,
        candidates: &'a [(String, f64, usize)],
        chars: &[char],
        pos: usize,
        total_len: usize,
    ) -> &'a (String, f64, usize) {
        // 如果有长词优先使用（符合中文分词的一般规律）
        let max_len = candidates.iter().map(|(_, _, len)| *len).max().unwrap();
        let longest_candidates: Vec<_> = candidates
            .iter()
            .filter(|(_, _, len)| *len == max_len)
            .collect();

        if longest_candidates.len() == 1 {
            return longest_candidates[0];
        }

        // 在相同长度的词中，选择词频最高的
        let mut best = longest_candidates[0];
        for candidate in longest_candidates.iter().skip(1) {
            // 优先选择词频更高的词
            if candidate.1 > best.1 {
                best = candidate;
            } else if candidate.1 == best.1 {
                // 词频相同的情况下，考虑后续匹配的可能性
                let next_pos = pos + candidate.2;
                let next_pos_best = pos + best.2;

                if next_pos < total_len {
                    let can_extend_next = self.can_extend(chars, next_pos);
                    let can_extend_best = self.can_extend(chars, next_pos_best);

                    if can_extend_next && !can_extend_best {
                        best = candidate;
                    }
                }
            }
        }

        best
    }

    // 检查在指定位置是否可以找到更长的匹配
    #[allow(dead_code)]
    fn can_extend(&self, chars: &[char], pos: usize) -> bool {
        if pos >= chars.len() {
            return false;
        }

        let mut node = &self.trie;
        #[allow(clippy::needless_range_loop)]
        for j in pos..chars.len().min(pos + self.max_word_len) {
            match node.get_child(chars[j]) {
                Some(child) => {
                    node = child;
                    if node.word_entry.is_some() {
                        return true;
                    }
                }
                None => break,
            }
        }
        false
    }

    // HMM 分词 (用于未登录词)
    fn cut_hmm(&self, sentence: &str) -> Vec<String> {
        let chars: Vec<char> = sentence.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }

        let states = self.hmm_model.viterbi(&chars);
        let mut result = Vec::new();
        let mut start = 0;

        for (i, &state) in states.iter().enumerate() {
            // 状态 2(E) 和 3(S) 表示一个词的结束
            if state == 2 || state == 3 {
                let word: String = chars[start..=i].iter().collect();
                result.push(word);
                start = i + 1;
            }
        }

        result
    }

    pub fn cut(&self, sentence: &str, hmm: bool) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }

        // 创建缓存键（包含sentence和hmm参数）
        let cache_key = format!("{}:{}", sentence, hmm);

        // 检查缓存
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached_result) = cache.get(&cache_key) {
                return cached_result.clone();
            }
        }

        // 直接使用正则表达式处理，这是最安全高效的方式
        let mut result = Vec::new();
        let mut start = 0;

        while start < sentence.len() {
            let remaining = &sentence[start..];

            // 查找中文字符序列
            if let Some(han_match) = self.re_han.find(remaining) {
                let han_start = han_match.start();
                let han_end = han_match.end();

                // 处理前面的非中文字符
                if han_start > 0 {
                    let non_han = &remaining[..han_start];
                    if let Some(skip_match) = self.re_skip.find(non_han) {
                        result.push(skip_match.as_str().to_string());
                        start += skip_match.end();
                    } else {
                        // 逐个处理非中文字符
                        for ch in non_han.chars() {
                            result.push(ch.to_string());
                        }
                        start += non_han.len();
                    }
                    continue;
                }

                // 处理中文字符
                let han_text = &remaining[han_start..han_end];

                // 使用算法进行分词
                let han_tokens = if hmm && han_text.chars().count() > 1 {
                    // 启用HMM：对未识别的连续字符使用HMM分词
                    self.cut_dag_with_hmm(han_text)
                } else {
                    // 不使用HMM：使用优化后的最大概率算法
                    self.cut_dag(han_text)
                };

                result.extend(han_tokens);
                start += han_end;
            } else {
                // 剩余全是非中文字符
                if let Some(skip_match) = self.re_skip.find(remaining) {
                    result.push(skip_match.as_str().to_string());
                    start += skip_match.end();
                } else {
                    // 逐个处理剩余字符
                    for ch in remaining.chars() {
                        result.push(ch.to_string());
                    }
                    break;
                }
            }
        }

        // 存储到缓存（如果缓存未满）
        {
            let mut cache = self.cache.write().unwrap();
            if cache.len() < self.max_cache_size {
                cache.insert(cache_key, result.clone());
            }
        }

        result
    }

    // 辅助方法：直接推送字符串到结果
    #[allow(dead_code)]
    fn push_str(&self, result: &mut Vec<String>, s: &str) {
        if s.is_empty() {
            return;
        }
        // 如果是单个字符且不在词典中，直接推送
        if s.chars().count() == 1 && !self.dict.contains_key(s) {
            result.push(s.to_string());
        } else {
            // 否则使用正则表达式处理
            if let Some(skip_match) = self.re_skip.find(s) {
                result.push(skip_match.as_str().to_string());
                let remaining = &s[skip_match.end()..];
                if !remaining.is_empty() {
                    self.push_str(result, remaining);
                }
            } else {
                for ch in s.chars() {
                    result.push(ch.to_string());
                }
            }
        }
    }

    // 添加动态词典管理方法
    pub fn add_word(&mut self, word: &str, freq: Option<f64>, pos: Option<&str>) {
        let freq = freq.unwrap_or(1.0);
        let pos = pos.unwrap_or("n").to_string();
        let log_freq = freq.ln();

        let word_entry = WordEntry {
            word: word.to_string(),
            freq,
            log_freq,
            pos,
        };

        // 更新最大词长
        if word.chars().count() > self.max_word_len {
            self.max_word_len = word.chars().count();
        }

        // 更新总词频
        if let Some(old_entry) = self.dict.get(word) {
            self.total_freq -= old_entry.freq;
        }
        self.total_freq += freq;

        // 添加到词典和 Trie 树
        self.dict.insert(word.to_string(), word_entry.clone());
        self.add_to_trie(word, word_entry);
    }

    pub fn del_word(&mut self, word: &str) -> bool {
        if let Some(entry) = self.dict.remove(word) {
            self.total_freq -= entry.freq;
            // Trie 树的删除比较复杂，这里简化处理
            true
        } else {
            false
        }
    }

    pub fn suggest_freq(&self, segment: &str, _tune: bool) -> f64 {
        let chars: Vec<char> = segment.chars().collect();
        let n = chars.len();

        if n == 0 {
            return 0.0;
        }

        let mut total_log_freq = 0.0;
        let mut _unknown_chars = 0;

        for ch in chars {
            let ch_str = ch.to_string();
            if let Some(entry) = self.dict.get(&ch_str) {
                total_log_freq += entry.freq.ln();
            } else {
                _unknown_chars += 1;
                total_log_freq += -3.14e+100; // 未知字符的极小概率
            }
        }

        let avg_log_freq = total_log_freq / n as f64;
        avg_log_freq.exp()
    }

    pub fn cut_for_search(&self, sentence: &str, hmm: bool) -> Vec<String> {
        let cut_result = self.cut(sentence, hmm);
        let mut result = Vec::new();

        for token in cut_result {
            if token.chars().count() > 2 {
                result.push(token.clone());

                for i in 1..token.chars().count() {
                    let chars: Vec<char> = token.chars().collect();
                    let sub_token: String = chars
                        .iter()
                        .skip(i)
                        .take(token.chars().count() - i)
                        .collect();
                    if sub_token.chars().count() > 1 && self.dict.contains_key(&sub_token) {
                        result.push(sub_token);
                    }
                }
            } else {
                result.push(token);
            }
        }

        result
    }

    pub fn cut_full(&self, sentence: &str, hmm: bool) -> Vec<String> {
        let cut_result = self.cut(sentence, hmm);
        let mut result = Vec::new();

        for token in cut_result {
            if token.chars().count() > 1 {
                self.generate_all_combinations(&token, &mut result);
            } else {
                result.push(token);
            }
        }

        result
    }

    fn generate_all_combinations(&self, word: &str, result: &mut Vec<String>) {
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();

        for i in 0..n {
            for j in (i + 1)..=n {
                let sub_word: String = chars.iter().skip(i).take(j - i).collect();
                if sub_word.chars().count() > 1 && self.dict.contains_key(&sub_word) {
                    result.push(sub_word);
                }
            }
        }

        result.push(word.to_string());
    }

    pub fn tokenize(
        &self,
        sentence: &str,
        mode: &str,
        hmm: bool,
    ) -> Vec<(String, String, usize, usize)> {
        let tokens = match mode {
            "search" => self.cut_for_search(sentence, hmm),
            "full" => self.cut_full(sentence, hmm),
            _ => self.cut(sentence, hmm),
        };

        let mut result = Vec::new();
        let mut offset = 0;

        for token in tokens {
            let start = offset;
            let end = offset + token.len();
            let pos = self
                .dict
                .get(&token)
                .map(|entry| entry.pos.clone())
                .unwrap_or_else(|| "n".to_string());
            result.push((token, pos, start, end));
            offset = end;
        }

        result
    }

    pub fn extract_tags(
        &self,
        sentence: &str,
        top_k: usize,
        allow_pos: Option<Vec<&str>>,
    ) -> Vec<(String, f64)> {
        let tokens = self.cut(sentence, true);
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0;

        for token in tokens {
            if token.chars().count() < 2 {
                continue;
            }

            if let Some(allow_pos) = &allow_pos {
                if let Some(entry) = self.dict.get(&token) {
                    if !allow_pos.contains(&entry.pos.as_str()) {
                        continue;
                    }
                }
            }

            *word_freq.entry(token).or_insert(0) += 1;
            total_words += 1;
        }

        let mut tfidf_scores: Vec<(String, f64)> = Vec::new();

        for (word, freq) in word_freq {
            let tf = freq as f64 / total_words as f64;
            let idf = (self.total_freq
                / (1.0 + self.dict.get(&word).map(|e| e.freq).unwrap_or(1.0)))
            .ln();
            let score = tf * idf;
            tfidf_scores.push((word, score));
        }

        tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        tfidf_scores.truncate(top_k);
        tfidf_scores
    }

    pub fn analyze_text(&self, sentence: &str) -> TextAnalysis {
        let tokens = self.cut(sentence, true);
        let mut word_count = 0;
        let char_count = sentence.chars().count();
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut pos_count: HashMap<String, usize> = HashMap::new();

        for token in tokens {
            word_count += 1;
            *word_freq.entry(token.clone()).or_insert(0) += 1;

            if let Some(entry) = self.dict.get(&token) {
                *pos_count.entry(entry.pos.clone()).or_insert(0) += 1;
            }
        }

        let keywords = self.extract_tags(sentence, 10, None);

        TextAnalysis {
            word_count,
            char_count,
            word_freq,
            pos_count,
            keywords,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextAnalysis {
    pub word_count: usize,
    pub char_count: usize,
    pub word_freq: HashMap<String, usize>,
    pub pos_count: HashMap<String, usize>,
    pub keywords: Vec<(String, f64)>,
}

static GLOBAL_JIEBA: Lazy<Arc<Jieba>> = Lazy::new(|| Arc::new(Jieba::new()));

#[allow(dead_code)]
//#[pyfunction]
//#[pyo3(signature = (sentence, cut_all=false, hmm=true))]
fn cut(
    sentence: &str,
    cut_all: bool,
    hmm: bool,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let tokens = if cut_all {
        GLOBAL_JIEBA.cut_full(sentence, hmm)
    } else {
        GLOBAL_JIEBA.cut(sentence, hmm)
    };

    Ok(tokens)
}

//#[pyfunction]
//#[pyo3(signature = (sentence, cut_all=false, hmm=true))]
#[allow(dead_code)]
fn lcut(
    sentence: &str,
    cut_all: bool,
    hmm: bool,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    cut(sentence, cut_all, hmm)
}

//#[pyfunction]
//#[pyo3(signature = (sentence, hmm=true))]
#[allow(dead_code)]
fn cut_for_search(sentence: &str, hmm: bool) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(GLOBAL_JIEBA.cut_for_search(sentence, hmm))
}

//#[pyfunction]
//#[pyo3(signature = (sentence, hmm=true))]
#[allow(dead_code)]
fn lcut_for_search(sentence: &str, hmm: bool) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    cut_for_search(sentence, hmm)
}

//#[pyfunction]
//#[pyo3(signature = (sentence, mode="default", hmm=true))]
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
fn tokenize(
    sentence: &str,
    mode: &str,
    hmm: bool,
)
 -> Result<Vec<(String, String, usize, usize)>, Box<dyn std::error::Error>> {
    Ok(GLOBAL_JIEBA.tokenize(sentence, mode, hmm))
}

//#[pyfunction]
#[allow(dead_code)]
fn load_userdict(dict_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut jieba = Jieba::new();
    jieba
        .load_dict(dict_path)
        .map_err(|e| format!("Failed to load dictionary: {}", e))?;
    Ok(())
}

// 词性标注接口
//#[pyfunction]
//#[pyo3(signature = (sentence, hmm=true))]
#[allow(dead_code)]
fn posseg_cut(
    sentence: &str,
    hmm: bool,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let tokens = GLOBAL_JIEBA.tokenize(sentence, "default", hmm);
    let result: Vec<(String, String)> = tokens
        .into_iter()
        .map(|(word, pos, _, _)| (word, pos))
        .collect();
    Ok(result)
}

// 动态词典管理接口
//#[pyfunction]
//#[pyo3(signature = (word, freq=None, tag=None))]
#[allow(dead_code)]
fn add_word(
    _word: &str,
    _freq: Option<f64>,
    _tag: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 由于 GLOBAL_JIEBA 是不可变的，我们需要使用全局可变状态
    // 这里简化处理，实际应该使用 Arc<RwLock<Jieba>>
    println!("Note: add_word functionality would require thread-safe global state");
    Ok(())
}

//#[pyfunction]
#[allow(dead_code)]
fn del_word(_word: &str) -> Result<bool, Box<dyn std::error::Error>> {
    // 同样需要线程安全的全局状态
    println!("Note: del_word functionality would require thread-safe global state");
    Ok(true)
}

//#[pyfunction]
//#[pyo3(signature = (segment, tune=true))]
#[allow(dead_code)]
fn suggest_freq(segment: &str, tune: bool) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(GLOBAL_JIEBA.suggest_freq(segment, tune))
}

#[pyclass]
struct JiebaTokenizer {
    jieba: Jieba,
}

#[pymethods]
impl JiebaTokenizer {
    #[new]
    #[pyo3(signature = (dict_path=None))]
    fn new(dict_path: Option<&str>) -> PyResult<Self> {
        let mut jieba = Jieba::new();
        if let Some(path) = dict_path {
            jieba.load_dict(path).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to load dictionary: {}",
                    e
                ))
            })?;
        }
        Ok(JiebaTokenizer { jieba })
    }

    #[pyo3(signature = (sentence, cut_all=false, hmm=true))]
    fn cut(&self, sentence: &str, cut_all: bool, hmm: bool) -> PyResult<Vec<String>> {
        let tokens = if cut_all {
            self.jieba.cut_full(sentence, hmm)
        } else {
            self.jieba.cut(sentence, hmm)
        };

        Ok(tokens)
    }

    #[pyo3(signature = (sentence, cut_all=false, hmm=true))]
    fn lcut(&self, sentence: &str, cut_all: bool, hmm: bool) -> PyResult<Vec<String>> {
        self.cut(sentence, cut_all, hmm)
    }

    #[pyo3(signature = (sentence, hmm=true))]
    fn cut_for_search(&self, sentence: &str, hmm: bool) -> PyResult<Vec<String>> {
        Ok(self.jieba.cut_for_search(sentence, hmm))
    }

    #[pyo3(signature = (sentence, hmm=true))]
    fn lcut_for_search(&self, sentence: &str, hmm: bool) -> PyResult<Vec<String>> {
        self.cut_for_search(sentence, hmm)
    }

    #[pyo3(signature = (sentence, mode="default", hmm=true))]
    fn tokenize(
        &self,
        sentence: &str,
        mode: &str,
        hmm: bool,
    ) -> PyResult<Vec<(String, String, usize, usize)>> {
        Ok(self.jieba.tokenize(sentence, mode, hmm))
    }

    #[pyo3(signature = (sentence, hmm=true))]
    fn posseg_cut(&self, sentence: &str, hmm: bool) -> PyResult<Vec<(String, String)>> {
        let tokens = self.jieba.tokenize(sentence, "default", hmm);
        let result: Vec<(String, String)> = tokens
            .into_iter()
            .map(|(word, pos, _, _)| (word, pos))
            .collect();
        Ok(result)
    }

    #[pyo3(signature = (sentence, top_k=20, allow_pos=None))]
    fn extract_tags(
        &self,
        sentence: &str,
        top_k: usize,
        allow_pos: Option<Vec<String>>,
    ) -> PyResult<Vec<(String, f64)>> {
        Ok(self.jieba.extract_tags(
            sentence,
            top_k,
            allow_pos
                .as_ref()
                .map(|pos| pos.iter().map(|s| s.as_str()).collect()),
        ))
    }

    fn analyze_text(&self, sentence: &str) -> PyResult<TextAnalysisPy> {
        let analysis = self.jieba.analyze_text(sentence);
        Ok(TextAnalysisPy {
            word_count: analysis.word_count,
            char_count: analysis.char_count,
            word_freq: analysis.word_freq,
            pos_count: analysis.pos_count,
            keywords: analysis.keywords,
        })
    }

    // 动态词典管理方法
    #[pyo3(signature = (word, freq=None, tag=None))]
    fn add_word(&mut self, word: &str, freq: Option<f64>, tag: Option<&str>) -> PyResult<()> {
        self.jieba.add_word(word, freq, tag);
        Ok(())
    }

    fn del_word(&mut self, word: &str) -> PyResult<bool> {
        Ok(self.jieba.del_word(word))
    }

    #[pyo3(signature = (segment, tune=true))]
    fn suggest_freq(&self, segment: &str, tune: bool) -> PyResult<f64> {
        Ok(self.jieba.suggest_freq(segment, tune))
    }

    fn load_userdict(&mut self, dict_path: &str) -> PyResult<()> {
        self.jieba.load_dict(dict_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load dictionary: {}", e))
        })?;
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
struct TextAnalysisPy {
    #[pyo3(get)]
    word_count: usize,
    #[pyo3(get)]
    char_count: usize,
    #[pyo3(get)]
    word_freq: HashMap<String, usize>,
    #[pyo3(get)]
    pos_count: HashMap<String, usize>,
    #[pyo3(get)]
    keywords: Vec<(String, f64)>,
}

//#[pyfunction]
//#[pyo3(signature = (sentence, top_k=20, allow_pos=None))]
#[allow(dead_code)]
fn extract_tags(
    sentence: &str,
    top_k: usize,
    allow_pos: Option<Vec<String>>,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    Ok(GLOBAL_JIEBA.extract_tags(
        sentence,
        top_k,
        allow_pos
            .as_ref()
            .map(|pos| pos.iter().map(|s| s.as_str()).collect()),
    ))
}

//#[pyfunction]
#[allow(dead_code)]
fn analyze_text(sentence: &str) -> Result<TextAnalysisPy, Box<dyn std::error::Error>> {
    let analysis = GLOBAL_JIEBA.analyze_text(sentence);
    Ok(TextAnalysisPy {
        word_count: analysis.word_count,
        char_count: analysis.char_count,
        word_freq: analysis.word_freq,
        pos_count: analysis.pos_count,
        keywords: analysis.keywords,
    })
}

// Test function for core functionality
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_segmentation() {
        let jieba = Jieba::new();
        let result = jieba.cut(
            "北京大学的计算机系学生正在研究自然语言处理和机器学习算法。",
            true,
        );
        println!("Segmentation result: {:?}", result);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_short_text() {
        let jieba = Jieba::new();
        let result = jieba.cut("我是一个学生", true);
        println!("Short text result: {:?}", result);
        assert!(!result.is_empty());
    }
}

// Python module definition
#[pymodule]
fn rust_jieba(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JiebaTokenizer>()?;
    m.add_class::<TextAnalysisPy>()?;
    Ok(())
}


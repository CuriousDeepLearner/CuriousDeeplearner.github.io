---
layout: post
comments: true
title:  "Language Model (Mô hình ngôn ngữ)"
title2:  "2. Language Model (Mô hình ngôn ngữ)"
date:   2017-11-24 15:22:00
permalink: 2017/11/24/nlp/
mathjax: true
tags: NLP
category: NLP, Language_model
sc_project: 11281831
sc_security: f2dfc7eb
img: \assets\2-language-model\Language-Models.png
summary: Language Model (Mô hình ngôn ngữ) là cở sở của rất nhiều mô hình xử lý ngôn ngữ về sau.
---

Trước khi đi vào các mô hình về word embeddings, thì chúng ta cùng đảo qua về Language Models (mô hình ngôn ngữ) - mô hình thường được sử dụng trước đây trong việc giải quyết các bài toán về nhận dạng tiếng nói (speech recognition), phiên dịch tự động (machine translation)... Việc hiểu được mô hình này sẽ giúp các bạn có những cơ sở kiến thức cơ bản về các mô hình phức tạp về sau như Hidden Markov Model, hay Long short-term memory (LSTM)

Đây là mô hình phân bố xác suất trên các tập văn bản. Cụ thể nó cho biết xác suất 1 câu (1 cụm từ hay 1 từ) trong bộ dữ liệu mẫu là bao nhiêu. Tức là, cung cấp thông tin về phân bố xác suất tiền nghiệm (prior distribution) \\(p(x_1, x_2,...x_n)\\) để xem xét 1 câu gồm 1 chuỗi các **từ đầu vào** có phù hợp hay không.    
Lấy ví dụ: sau khi thực hiện Language Modelling ta sẽ có     

$$ p(\text{học xử lý ngôn ngữ tự nhiên}) = 0,001 > p(\text{tự lý xử nhiên ngôn học ngữ}) = 0  $$  

Nhờ vậy chúng ta sẽ xác định được câu "học xử lý ngôn ngữ tự nhiên" sẽ phù hợp hơn với ngôn ngữ tiếng viêt hơn câu "tự lý xử nhiên ngôn học ngữ".

### Bài toán:    
Giả sử chúng ta có 1 tập ngữ liệu (corpus) tiếng việt thu thập được từ các trang web và 1 từ điển \\( V \\). Ở đây, ngữ liệu là tập dữ liệu gồm các câu (sentence) cho một ngôn ngữ xác đinh, ở đây là tiếng việt; từ điển \\( V \\) là 1 bộ từ vựng gồm tập hợp hữu hạn các từ có độ dài là \\(|V|\\).      
Xem xét 1 câu A bất kỳ gồm n từ bao gồm 1 chuỗi các từ \\( x_1, x_2,...x_n \\) nằm trong bộ từ điển \\( V \\) ban đầu

Mục tiêu của chúng ta là xây dựng 1 mô hình có khả năng tính toán xác suất của câu này thuộc về ngôn ngữ mà chúng ta đang xem xét \\(p(x_1, x_2,...x_n)\\)

### 1. Cách tiếp cận đơn giản
Cách đầu tiên chúng ta có thể nghĩ đến chính là sử dụng việc đếm. Đơn giản bằng cách đếm số lần câu A của chúng ta xuất hiện bao nhiêu lần trong ngữ liệu (corpus) chia cho số lượng câu có trong tập ngữ liệu huẩn luyện.

Trình bày toán học:     
Ký hiệu \\( V^{+}\\) là tập hợp tất cả các câu khởi tạo từ bộ từ vựng \\(V\\).     
Một câu bao gồm n từ có dạng \\(x_1x_2x_3...x_n\\)     
với \\( n \ge 1 \\), \\(x_i \in V \\) và \\(x_n = STOP \\) là 1 ký hiệu đặc biệt \\( STOP \not \in V \\)     
Một mô hình gồm tập hữu hạn từ vựng \\(V\\) và 1 hàm xác suất \\(p(x_1, x_2,...x_n)\\) sao cho:     

$$p(x_1,x_2,...,x_n) \ge  0 ~~ x_i \in V^+ , i = 1,2,...n$$          
$$\sum_{<x_1...x_n> \in V^+} p(x_1,x_2,...,x_n) = 1$$

Như vậy \\(p(x_1, x_2,...x_n)\\) là phân bố xác suất của các câu trong tập \\(V^+\\)

Một cách đơn giản ta có thể tính xác suất trên như sau:    

$$p(x_1...x_n) = \frac{c(x_1...x_n)}{N} $$

Với $c(x_1...x_n)$ (count) là số lần xuất hiện của câu \\(x_1...x_n\\) trong ngữ liệu, và \\(N\\) là số lượng các câu trong ngữ liệu huấn luyện.

**Nhược điểm:** Điểm trừ lớn nhất của phương pháp này chính là không có khả năng tổng quát hóa (generalisation). Lấy ví dụ nếu 1 câu mới không có trong tập ngữ liệu corpus thì tử số của xác suất sẽ bằng 0, trong khi điều ta mong muốn là có thể dự đoán được bất kỳ câu mới nào. Chúng ta sẽ cùng xem xét những phương pháp tiếp theo để khắc phục nhược điểm này

Ví dụ: giả sử mình có 1 bộ từ điển như sau (trên thực tế bộ từ điển có thể bao gồm hàng nghìn thậm chí trăm ngàn từ):    

 $$ V = \text{\{tôi, là, Tú, không, thích, hành, và, tỏi\}}$$

### 2. Markov Models
Ý tưởng chính của mô hình Markov là giả định xác suất của đối tượng (câu, từ hoặc cụm từ đang nghiên cứu) chỉ phụ thuộc vào \\(k \(k \leq n- 1\)\\) đối tượng trước đó của một chuỗi.

Xét một chuỗi các biến ngẫu nhiên \\(X_1, X_2, ..., X_n\\). Mỗi biến ngẫu nhiên này có thể có các giá trị thuộc tập hữu hạn V. Mục tiêu của chúng ta là mô hình hoá xác suất của một chuỗi bất kỳ \\(x_1x_2...x_n\\), trong đó \\(n \ge 1\\) và \\(x_i \in V\\) sao cho \\(i=1,...,n\\) như sau:

$$P(X_1=x_1, X_2=x_2,...,X_n=x_n)$$   
Ký hiệu đơn giản như sau \\(P(x_1x_2x_3....x_n)\\)
Ở đây chúng ta cần sự trợ giúp của 1 công cụ rất nổi tiếng trong thống kê **"chain rule" hay công thức Bayes**: \\(P(AB) = P(B|A) . P(A)\\)       
Từ đó ta có công thức sau:     

\\[
\begin{eqnarray}
    P(X_1=x_1, X_2=x_2,..., X_n=x_n) =& P(X_1=x_1).P(X_2=x_2|X_1=x_1).P(X_3=x_3|X_1=x_1,X_2=x_2).... ~~~~ \\\
    =& P(X_1=x_1)\prod_{i=2}^n P(X_i=x_i|X_1=x_1,...,X_{i-1}=x_{i-1}) \quad (1.1)
\end{eqnarray}
\\]


Viết đơn giản hơn ta có:    

$$ P(x_1x_2x_3....x_n) = P(x_1).\prod_{i=2}^n P(x_i|x_1x_2....x_{i-1})$$

#### 2.1 Mô hình Markov bậc 1
Giả sử \\(x_i\\) chỉ phụ thuộc vào điều kiện của 1 đối tượng trước đó là \\(x_{i-1}\\) trong chuỗi. Nghĩa là:   

$$P(X_i=x_i|X_1=x_1,...,X_{i-1}=x_{i-1}) = P(X_i=x_i|X_{i-1}=x_{i-1}) \quad (1.2)$$

Kết hợp (1.1) và (1.2) ta thu được:      

$$P(X_1=x_1, X_2=x_2,..., X_n=x_n) = P(X_1=x_1)\prod_{i=2}^n P(X_i=x_i|X_{i-1}=x_{i-1}) $$

Viết đơn giản hơn:   

$$ P(x_1x_2...x_n)  = P(x_1)\prod_{i=2}^n P(x_i|x_{i-1}) $$


#### 2.2 Mô hình Markov bậc k
Tương tự như mô hình bậc 1, lần này ta giả định rằng đối tượng thứ \\(i\\) phụ thuộc vào \\(k\\) đối tượng trước nó.   

$$ P(x_1x_2...x_n)  = P(x_1)\prod_{i=2}^n P(x_i|x_{i-1}x_{i-2}...x_{i-k}) $$



### 2.Mô hình ngôn ngữ N-gram     
#### 2.1 Một số khái niệm
**Ngữ liệu**
Ngữ liệu (corpus) là 1 dữ liệu tập hợp các văn bản, ngôn ngữ đã được số hóa. Ví dụ về corpus như "tuyển tập truyện ngắn Vũ Trọng Phụng" hay "tuyển tập các bài hát nhạc vàng"

**N-gram**    
Là tần suất xuất hiện của n kí tự (hoặc từ) liên tiếp nhau có trong dữ liệu của corpus     
- Với n = 1: unigram, và tính trên kí tự, ta có thông tin về tần suất xuất hiện nhiều nhất của các chữ cái.     
- n = 2 : bigram.

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
img: \assets\2-language-model\Language-Model.png
summary: Language Model (Mô hình ngôn ngữ) là cở sở của rất nhiều mô hình xử lý ngôn ngữ về sau.
---

Trước khi đi vào các mô hình về word embeddings, thì chúng ta cùng đảo qua về Language Models - mô hình thường được sử dụng trước đây trong việc giải quyết các bài toán về nhận dạng tiếng nói (speech recognition), phiên dịch tự động (machine translation)... Việc hiểu được mô hình này sẽ giúp các bạn có những cơ sở kiến thức cơ bản về các mô hình phức tạp về sau như Hidden Markov Model, hay Long short-term memory (LSTM)

Đây là mô hình cung cấp thông tin về phân bố xác suất tiền nghiệm (prior distribution) \\(p(x_1, x_2,...x_n)\\) để xem xét 1 câu gồm 1 chuỗi các **từ đầu vào** có phù hợp hay không.    
Lấy ví dụ: sau khi thực hiện Language Modelling ta sẽ có     
\\[
\begin{eqnarray}
\\(p(học xử lý ngôn ngữ tự nhiên thật vui) > p(tự lý xử nhiên vui thật ngôn ngữ học) \\)  
\end{eqnarray}
\\]
Nhờ vậy chúng ta sẽ xác định được câu "học xử lý ngôn ngữ tự nhiên thật vui" sẽ phù hợp hơn với ngôn ngữ tiếng viêt hơn câu "tự lý xử nhiên vui thật ngôn ngữ học".

#### Bài toán:    
Giả sử chúng ta có 1 tập ngữ liệu (corpus) tiếng việt thu thập được từ các trang web và 1 từ điển \\( V \\). Ở đây, ngữ liệu là tập dữ liệu gồm các câu (sentence) cho một ngôn ngữ xác đinh, ở đây là tiếng việt; từ điển \\( V \\) là 1 bộ từ vựng gồm tập hợp hữu hạn các từ có độ dài là \\(|V|\\).      
Xem xét 1 câu A bất kỳ gồm n từ bao gồm 1 chuỗi các từ \\( x_1, x_2,...x_n \\) nằm trong bộ từ điển \\( V \\) ban đầu

Mục tiêu của chúng ta là xây dựng 1 mô hình có khả năng tính toán xác suất của câu này thuộc về ngôn ngữ mà chúng ta đang xem xét \\(p(x_1, x_2,...x_n)\\)

Cách đầu tiên chúng ta có thể nghĩ đến chính là sử dụng việc đếm. Đơn giản bằng cách đếm số lần câu A của chúng ta xuất hiện bao nhiêu lần trong ngữ liệu (corpus) chia cho số lượng câu có trong tập ngữ liệu huẩn luyện.

Trình bày toán học:     
Ký hiệu \\( V^{+}\\) là tập hợp tất cả các câu khởi tạo từ bộ từ vựng \\(V\\).     
Một câu bao gồm n từ có dạng \\(x_1x_2x_3...x_n\\)     
với \\( n >= 1 \\), \\(x_i \in V \\) và \\(x_n = STOP \\)

Ví dụ: giả sử mình có 1 bộ từ điển như sau (trên thực tế bộ từ điển có thể bao gồm hàng nghìn thậm chí trăm ngàn từ):    
\\[
\begin{eqnarray}
\\( V = {tôi, là, Hùng, không, thích, hành, và, tỏi })
\end{eqnarray}
\\]

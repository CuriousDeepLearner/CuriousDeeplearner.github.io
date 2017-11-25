---
layout: post
comments: true
title:  "Tổng quan về Natural Language Preprocessing (NLP)"
title2:  "1. Tổng quan về Natural Language Preprocessing (NLP)"
date:   2017-11-23 15:22:00
permalink: 2017/11/23/nlp/
mathjax: true
tags: NLP
category: NLP
sc_project: 11281831
sc_security: f2dfc7eb
img: \assets\NLP_intro\NLP_intro.png
summary: Tổng quan về Natural Language Preprocessing (NLP)
---
*Note: trong blog này tôi vẫn sẽ giữ nguyên các thuật ngữ chuyên ngành tiếng anh (bên cạnh có dịch sơ lược sang tiếng việt). Tôi cũng khuyến khích bạn đọc nên sử dụng các thuật ngữ này khi nói hay làm việc, để có sự thống nhất đồng bộ và tránh những nhầm lẫn không đáng có. Ví dụ: bạn thường nên dùng thuật ngữ "Machine Learning" hơn là sử dụng "máy học"*

**Natural Language Preprocessing (NLP)** (dịch sang tiếng việt là **"Xử lý ngôn ngữ tự nhiên"**) là một mảng đang nhận được rất nhiều sự chú ý trong giới khoa học về Machine Learning (Máy học) gần đây, đặc biệt là từ năm 2012 với sự ra đời của **word2vec**.  

### Ứng dụng của NLP trong thực tế:

### Các thuật ngữ cơ bản:
1. Đầu tiên chúng ta sẽ cùng làm quen với 1 thuật ngữ cơ bản nhất trong NLP là **word embedding** (tạm dịch ra là **nhúng từ**): Đây là kỹ thuật biến đổi *1 từ* hay *câu* có trong *từ điển* sang dạng *vectors số*. Những vectors *gần giống nhau* sẽ biểu thị những từ với ý nghĩa gần giống nhau.          
Hiểu đơn giản là ngôn ngữ bình thường con người sử dụng là dạng *từ* hay *câu* trong khi ngôn ngữ của máy tính là dạng *số*. Vậy thì để máy tính có thể hiểu được ngôn ngữ con người, cần 1 cách thức có thể chuyển ngôn ngữ người sang ngôn ngữ máy và ngược lại. Trong toán học, **embedding** ( dịch là **nhúng**) dùng để chỉ 1 hàm hay 1 cách biến đổi 1 tập hợp X sang tập hợp Y -- hay ta gọi nhúng X vào Y.

2. **Word vectors** (**distributed representations**): Vector của từ

3. **Dimensional space**: không gian nhiều chiều

4. **computational complexity**: đây là 1 thuật ngữ thường bắt gặp trong ngành Machine learning nói chung và Deep Learning nói riêng, để chỉ độ phức tạp tính toán của mô hình. Mô hình càng phức tạp thì càng mất nhiều thời gian và tài nguyên để chạy.

### Lịch sử của word embeddings
- Kể từ những năm 90, mô hình không gian vector đã được sử dụng trong việc phân phối nghĩa của từ. Nhiều mô hình word embeddings đã được phát triển, nổi bật trong đó là **Latent Semantic Analysis (LSA)** và **Latent Dirichlet Allocation(LDA)**

### Các mô hình word embeddings
Về cơ bản, các mô hình hiện tại:     
+> sử dụng **từ** trong **từ điển** như là **đầu vào**     
+> biến chúng thành những vectors trong không gian với chiều thấp hơn.      
+> sau đó thay đổi (fine-tune) weights, các tham số thông qua **back-propagation**, để tạo thành **Embedding Layer (Lớp nhúng)**      

Sự khác nhau cơ bản giữa các mô hình này và với mô hình word2vec mà chúng ta sẽ nghiên cứu trong bài sau là về vấn đề độ phức tạp trong tính toán (*computational complexity*). Việc sử dụng hệ thống kiến trúc quá sâu của các lớp hay quá phức tạp sẽ khiến cho mô hình ngốn nhiều tài nguyên và thời gian hơn đặc biệt trong trường hợp số lượng từ có trong từ điển quá lớn. Đó là lý do chính tại sao mãi đến năm 2013, chúng ta mới nhìn thấy những thành tựu lớn trong mảng NLP khi mà các vi xử lý đã được cải thiện hơn rất nhiều so với trước đây.      

#### Bài toán:    
Giả sử chúng ta có 1 đoạn text training chứa T từ $w_1, w_2, ..., w_T$ nằm trong 1 từ điển V có độ dài |V|.     
Mô hình của chúng ta xem xét 1 bối cảnh gồm n từ, biểu diễn mỗi từ đầu vào dưới dạng vector *input embedding* $v_m$ với $d$ chiều, và đầu ra *output embedding* $v_m'$ bằng cách tối ưu hóa hàm mất mát $J_{\{phi}}$ với tham số mô hình là $\phi$.      




This post is credited to the course: [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

#include "../../hnswlib/hnswlib.h"


int main() {
    /******** 1. 基本参数 ********/
    const int dim          = 128;               // 向量维度
    const size_t max_elems = 1000000;         // 建库时的 max_elements_
    const std::string file = "/home/cheng_zou/cuvs/my_index.bin"; // 索引文件

    /******** 2. 创建空 Index 对象 ********/
    // space='l2' / 'ip' / 'cosine'
    hnswlib::L2Space space(dim);                // L2 (平方欧氏)
    // label type = size_t, data type = float
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, file);

    /******** 3. 加载索引 ********/
    // loadIndex 会覆盖 index 对象内部数据结构
    // index.loadIndex(file, max_elems);

    return 0;
}

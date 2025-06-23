#include "../../hnswlib/hnswlib.h"
#include <chrono>

uint32_t get_arg(char *arg) {
    uint32_t num_arg = 0;
    for(int i = 0; i < strlen(arg); i++) {
        assert(arg[i] >= '0' && arg[i] <= '9');
        num_arg = num_arg * 10 + arg[i] - '0';
    }
    return num_arg;
} 

int main(int argc, char **argv) {
    assert(argc == 2);
    char *arg = argv[1];
    int K=10;

    unsigned d_xq, d_xt;
    unsigned m_xq, m_xt;
    std::ios::pos_type ss;
    size_t fsize;

    std::ifstream in_query("/Users/skytbac/Documents/PIM_RAG/sift1M/sift_query.fvecs", std::ios::binary);
    if(!in_query.is_open()) {
        printf("Open file error");
        exit(-1);
    }
    in_query.read((char*)&d_xq, 4);
    in_query.seekg(0, std::ios::end);
    ss = in_query.tellg();
    fsize = (size_t)ss;
    m_xq = (unsigned)(fsize / (d_xq + 1) / 4);
    std::vector<float> data_query((size_t)m_xq * (size_t)d_xq);
    in_query.seekg(0, std::ios::beg);
    for(size_t i = 0; i < m_xq; i++) {
        in_query.seekg(4, std::ios::cur);
        in_query.read((char*)&data_query[i * d_xq], d_xq * 4);

        // normalize the query vector to [0, 1]
        for(size_t j = 0; j < d_xq; j++) {
            data_query[i * d_xq + j] = data_query[i * d_xq + j] / 255.0f; // normalize to [0, 1]
        }
    }


    std::ifstream in_gt("/Users/skytbac/Documents/PIM_RAG/sift1M/sift_groundtruth.ivecs", std::ios::binary);
    if(!in_gt.is_open()) {
        printf("Open file error");
        exit(-1);
    }
    in_gt.read((char*)&d_xt, 4);
    in_gt.seekg(0, std::ios::end);
    ss = in_gt.tellg();
    fsize = (size_t)ss;
    m_xt = (unsigned)(fsize / (d_xt + 1) / 4);
    std::vector<int> data_gt((size_t)m_xt * (size_t)d_xt);
    in_gt.seekg(0, std::ios::beg);
    for(size_t i = 0; i < m_xt; i++) {
        in_gt.seekg(4, std::ios::cur);
        in_gt.read((char*)&data_gt[i * d_xt], d_xt * 4);
    }    

    int dim = 128;
    int max_elements = 1000000;
    std::string hnsw_path = "/Users/skytbac/Downloads/TEMP_FILE/sift1M_CPU_cagra.bin";

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw;
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);

    alg_hnsw->setEf(get_arg(arg));
    alg_hnsw->metric_hops = 0;
    alg_hnsw->metric_distance_computations = 0;

    int success = 0;
    std::cout << "query num: " << m_xq << " total_level: " << alg_hnsw->maxlevel_ << " ef: " << get_arg(arg) << std::endl;
    
    for(int i = 0; i < 1; i++) {
    // for(int i = 0; i < 1; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_query.data() + (i * d_xq), K);
        for(int j = 0; j < K; j++) {
            for(int k = 0; k < K; k++) if(result.top().second == data_gt[k + i * d_xt]) success++;
            std::cout << result.top().second << "/" << result.top().first << std::endl;
            result.pop();
        }
        // std::cout << std::endl;
    }
    
    // std::cout << "QPS: " << m_xq / search_time << std::endl;
    std::cout << "dist_op_ns_: " << alg_hnsw->dist_op_ns_ << std::endl;
    std::cout << "data_bytes: " << alg_hnsw->dist_ops_/2*4 << std::endl;
    std::cout << "dist_OPS: " << alg_hnsw->dist_ops_ << std::endl;
    std::cout << "result_cmp_cnt_: " << alg_hnsw->result_cmp_cnt_ << std::endl;
    std::cout << "topk_cmp_cnt_: " << alg_hnsw->topk_cmp_cnt_ << std::endl;
    std::cout << "candidate_cmp_cnt_: " << alg_hnsw->candidate_cmp_cnt_ << std::endl;
    uint64_t total_ops = alg_hnsw->dist_ops_ + 
                         alg_hnsw->result_cmp_cnt_ +
                         alg_hnsw->topk_cmp_cnt_ + 
                         alg_hnsw->candidate_cmp_cnt_;
    std::cout << "total_ops: " << total_ops << std::endl;
    std::cout << "recall10@"<<K<<": " << 1.0*success / (m_xq * K)*100 << std::endl;
    std::cout << "dis_compute per query: " << alg_hnsw->metric_distance_computations / (m_xq * 1.0) << std::endl;
    std::cout << "hops per query: " << alg_hnsw->metric_hops / (m_xq * 1.0) << std::endl;
}
#include "../Dense-CSR/csr_matrix.h"
#include "../Dense-CSR/vector_operations.h"
#include <cmath>

std::vector<double> simple(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const double &t, const size_t &N, const double &tolerance){
    std::vector<double> x = x0;
    std::vector<double> r = std::vector<double>(b.size());
    double diff = 0.;
    for(size_t i = 0; i < N; i++){
        r = (A*x - b);
        x = x - t*r;
        diff = 0;
        for(size_t j = 0; j < b.size(); j++){
            diff += (t * r[j]) * (t * r[j]);
        }
        if(std::sqrt(diff) < tolerance){
            break;
        }
    }
    return x;
}

std::vector<double> jacobi(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const double &t, const size_t &N, const double &tolerance){
    std::vector<double> x = x0;
    std::vector<double> tmp = std::vector<double>(b.size());
    double diff = 0.;
    for(size_t i = 0; i < N; i++){
        tmp = b - A*x;
        diff = 0;
        for(size_t j = 0; j < b.size(); j++){
            tmp[j] = (tmp[j] + x[j] * A(j, j)) / A(j, j);
            diff += (tmp[j] - x[j]) * (tmp[j] - x[j]);
        }
        if(std::sqrt(diff) < tolerance){
            x = tmp;
            break;
        }
        x = tmp;
    }
    return x;
}

std::vector<double> gauss_seidel(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const double &t, const size_t &N, const double &tolerance){
    std::vector<double> x = x0;
    std::vector<double> tmp = std::vector<double>(b.size());
    double diff = 0.;
    for(size_t i = 0; i < N; i++){
        for(size_t m = 0; m < b.size(); m++){
            tmp[m] = 0;
            for(size_t k = m + 1; k < b.size(); k++){
                tmp[m] += A(m, k) * x[k];
            }
        }
        for(size_t m = 0; m < b.size(); m++){
            for(size_t k = 0; k < m; k++){
                tmp[m] = A(m, k) * x[k];
            }
        }
        diff = 0;
        for(size_t j = 0; j < b.size(); j++){
            tmp[j] = (b[j] - tmp[j]) / A(j, j);
            diff += (tmp[j] - x[j]) * (tmp[j] - x[j]);
        }
        if(std::sqrt(diff) < tolerance){
            x = tmp;
            break;
        }
        x = tmp;
    }
    return x;
}
#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>


/// Count CPU ticks using rdtsc
uint64_t count_ticks() {
    uint32_t lo, hi;
    asm volatile ("rdtsc\n" : "=a" (lo), "=d" (hi));
    return ((uint64_t) hi << 32) | lo;
}

void std_benchmark(const std::vector<int> &first, const std::vector<int> &second) {
    uint64_t vector_size = first.size();

    // Must not use rdx:rax because they are used by rdtsc
    register double std1 asm("%r8");
    register int std2 asm("%r9");
    register double std3 asm("%r10");

    uint64_t total_ticks_dependent = 0, total_ticks_independent = 0;

    for (uint64_t i = 0; i < vector_size; ++i) {
        std1 = static_cast<double>(first[i]);
        std2 = second[i];

        // Dependent operations
        uint64_t start = count_ticks();
        std1 = std1 + static_cast<double>(std2);
        uint64_t end = count_ticks();
        total_ticks_dependent += end - start;

        // Independent operations
        start = count_ticks();
        std3 = std1 + static_cast<double>(std2);
        end = count_ticks();
        total_ticks_independent += end - start;
    }
    std::cout << "STD:" << '\n'
              << " Latency: " << (double) total_ticks_dependent / vector_size << " ticks ("
              << total_ticks_dependent << " ticks per " << vector_size << " operations)\n"
              << " Result issue rate: " << (double) total_ticks_independent / vector_size << " ticks ("
              << total_ticks_independent << " ticks per " << vector_size << " operations)\n\n";
}

void sse2_benchmark(const std::vector<int> &first, const std::vector<int> &second) {
    uint64_t block_size = 4;
    uint64_t vector_size = first.size() / block_size;

    __m128d sse1;
    __m128i sse2;
    __m128d sse3;

    int64_t total_ticks_dependent = 0, total_ticks_independent = 0;

    for (uint64_t i = 0; i < vector_size; i += block_size)
    {
        sse1 = _mm_cvtepi32_pd(_mm_loadu_si128((__m128i *) &first[i]));
        sse2 = _mm_loadu_si128((__m128i *) &second[i]);

        // Dependent operations
        uint64_t start = count_ticks();
        sse1 = _mm_add_pd(sse1, _mm_castsi128_pd(sse2));
        uint64_t end = count_ticks();
        total_ticks_dependent += end - start;

        // Independent operations
        start = count_ticks();
        sse3 = _mm_add_pd(sse1, _mm_castsi128_pd(sse2));
        end = count_ticks();
        total_ticks_independent += end - start;
    }
    // Scaling execution time for accurate comparison
    total_ticks_dependent *= block_size;
    total_ticks_independent *= block_size;
    std::cout << "SSE2:" << '\n'
              << " Latency: " << (double) total_ticks_dependent * block_size / vector_size << " ticks ("
              << total_ticks_dependent << " ticks per " << vector_size << " operations)\n"
              << " Result issue rate: " << (double) total_ticks_independent * block_size / vector_size << " ticks ("
              << total_ticks_independent << " ticks per " << vector_size << " operations)\n\n";
}

void avx2_benchmark(const std::vector<int> &first, const std::vector<int> &second) {
    uint64_t block_size = 8;
    uint64_t vector_size = first.size() / block_size;

    __m256d avx1;
    __m256i avx2;
    __m256d avx3;

    uint64_t total_ticks_dependent = 0, total_ticks_independent = 0;

    for (uint64_t i = 0; i < vector_size; i += block_size)
    {
        avx1 = _mm256_castsi256_pd(_mm256_loadu_si256((__m256i *) &first[i]));
        avx2 = _mm256_loadu_si256((__m256i *) &second[i]);

        // Dependent operations
        uint64_t start = count_ticks();
        avx1 = _mm256_add_pd(avx1, _mm256_castsi256_pd(avx2));
        uint64_t end = count_ticks();
        total_ticks_dependent += end - start;

        // Independent operations
        start = count_ticks();
        avx3 = _mm256_add_pd(avx1, _mm256_castsi256_pd(avx2));
        end = count_ticks();
        total_ticks_independent += end - start;
    }
    // Scaling execution time for accurate comparison
    total_ticks_dependent *= block_size;
    total_ticks_independent *= block_size;
    std::cout << "AVX2:" << '\n'
              << " Latency: " << (double) total_ticks_dependent * block_size / vector_size << " ticks ("
              << total_ticks_dependent << " ticks per " << vector_size << " operations)\n"
              << " Result issue rate: " << (double) total_ticks_independent * block_size / vector_size << " ticks ("
              << total_ticks_independent << " ticks per " << vector_size << " operations)\n\n";
}

int main() {
    std::size_t size = 1000;
    std::vector<int> first(size), second(size);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(INT32_MIN, INT32_MAX);
    for (std::size_t i = 0; i < size; ++i) {
        first[i] = distribution(generator);
        second[i] = distribution(generator);
    }

    std_benchmark(first, second);
    sse2_benchmark(first, second);
    avx2_benchmark(first, second);

    return EXIT_SUCCESS;
}

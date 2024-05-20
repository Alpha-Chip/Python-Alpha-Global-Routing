#ifndef __CPU_DISJOIN_SET_HPP__
#define __CPU_DISJOIN_SET_HPP__

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

/**
 * @class CpuDisjointSet
 * @brief Implements a disjoint-set data structure.
 *
 * This class provides an implementation of the disjoint-set (or union-find) data structure.
 * It is used to keep track of a set of elements partitioned into a number of disjoint (non-overlapping) subsets.
 */
class CpuDisjointSet {
public:
    CpuDisjointSet() = default;
    ~CpuDisjointSet() = default;

    /**
     * @brief Creates a new set with a single element.
     * @param t_set The element to be added as a new set.
     *
     * If the element is already part of a set, this function does nothing.
     */
    void make_set(int32_t t_set)
    {
        if (m_parent.find(t_set) != m_parent.end()) {
            throw std::runtime_error("Trying to add existing disjoin set " + std::to_string(t_set) + " as new one!");
        }

        m_parent.emplace(std::make_pair(t_set, t_set));
        m_rank.emplace(std::make_pair(t_set, 0));
        ++m_num_sets;
    }

    /**
     * @brief Finds the representative of the set that an element is a part of.
     *
     * @param t_k The element whose set representative is to be found.
     * @return The representative of the set, or -1 if the element is not found in any set.
     */
    int32_t find(int32_t t_k)
    {
        if (m_parent.find(t_k) == m_parent.end()) {
            throw std::runtime_error("Trying to find root of nonexisting disjoin set: " + std::to_string(t_k) + "!");
        }

        if (m_parent[t_k] != t_k) {
            m_parent[t_k] = find(m_parent[t_k]);
        }

        return m_parent[t_k];
    }

    /**
     * @brief Merges two sets represented by two elements.
     *
     * @param t_a An element of the first set.
     * @param t_b An element of the second set.
     *
     * If either element is not part of any set or if both elements are in the same set,
     * the function does nothing.
     */
    void union_sets(int32_t t_a, int32_t t_b)
    {
        int32_t x = find(t_a);
        int32_t y = find(t_b);

        if (x == y) {
            return;
        }

        if (m_rank[x] < m_rank[y]) {
            m_parent[x] = y;
        } else if (m_rank[x] > m_rank[y]) {
            m_parent[y] = x;
        } else {
            m_parent[y] = x;
            ++m_rank[x];
        }

        --m_num_sets;
    }

    /**
     * @brief Checks if all sets is in one set.
     *
     * @return bool true if all sets is in one set.
     */
    bool is_one_set() const noexcept { return m_num_sets == 1; }

    /**
     * @brief Clears the disjoint set, removing all elements and sets.
     */
    void clear() noexcept
    {
        m_parent.clear();
        m_rank.clear();
        m_num_sets = 0;
    }

private:
    std::unordered_map<int32_t, int32_t> m_parent {}; ///< Stores the parent of each element.
    std::unordered_map<int32_t, int32_t> m_rank {}; ///< Stores the rank of each set.
    int32_t m_num_sets {}; ///> Total number of disjoint sets.
};

#endif
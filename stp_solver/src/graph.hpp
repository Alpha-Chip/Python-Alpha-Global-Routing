#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @class Edge
 * @brief Class representing an edge in a graph.
 *
 * This class represents an edge with a weight, source, and destination.
 * It provides methods to get the weight, source, and destination of the edge,
 * as well as an overload of the greater-than operator for comparing edges.
 */
class Edge {
public:
    /**
     * @brief Construct a new Edge object.
     *
     * @param t_weight Weight of the edge.
     * @param t_source Source vertex of the edge.
     * @param t_destination Destination vertex of the edge.
     */
    explicit Edge(double t_weight, int32_t t_source, int32_t t_destination, int32_t t_prev1 = -1, int32_t t_prev2 = -1)
        : m_weight(t_weight)
        , m_source(t_source)
        , m_destination(t_destination)
        , m_prev_source(t_prev1)
        , m_prev_destination(t_prev2) {};

    ~Edge() = default;

    /**
     * @brief Get the weight of the edge.
     *
     * @return Value of the weight of the edge.
     */
    double get_weight() const noexcept { return m_weight; };

    /**
     * @brief Get the source vertex of the edge.
     *
     * @return Value of the source vertex of the edge.
     */
    int32_t get_source() const noexcept { return m_source; };

    /**
     * @brief Get the destination vertex of the edge.
     *
     * @return Value of the destination vertex of the edge.
     */
    int32_t get_destination() const noexcept { return m_destination; };

    /**
     * @brief Get the previous source vertex of the edge.
     *
     * @return Reference to the previous source vertex of the edge.
     */
    int32_t get_prev_source() const noexcept { return m_prev_source; };

    /**
     * @brief Get the previous destination vertex of the edge.
     *
     * @return Reference to the previous destination vertex of the edge.
     */
    int32_t get_prev_destination() const noexcept { return m_prev_destination; };

    /**
     * @brief Compare this edge with another edge based on weight.
     *
     * @param t_lhs The edge to compare with.
     * @return True if this edge's weight is greater than the other edge's weight.
     */
    bool operator>(const Edge& t_lhs) const noexcept { return m_weight > t_lhs.get_weight(); }

private:
    double m_weight {}; ///> Weight of the edge.
    int32_t m_source {}; ///> Source vertex of the edge.
    int32_t m_destination {}; ///> Destination vertex of the edge.
    int32_t m_prev_source {}; ///> Previous source for this edge
    int32_t m_prev_destination {}; ///> Previous destination for this edge
};

/**
 * @struct AdjacencyList
 * @brief A structure to represent an adjacency list of a graph.
 *
 * This structure provides an efficient representation of a graph using an adjacency list.
 * It stores a list of edges and a mapping of node IDs to their corresponding edges.
 */
struct AdjacencyList {
    /**
     * @brief List of all edges in the graph.
     */
    std::vector<Edge> edges {};

    /**
     * @brief Map of node IDs to their corresponding list of edges.
     */
    std::vector<std::vector<Edge*>> nodes {};

    /**
     * @brief Finds an edge between two specified nodes.
     *
     * @param t_a The ID of the first node.
     * @param t_b The ID of the second node.
     * @return An iterator to the found Edge in the adjacency list. If no edge is found, it returns a const iterator to the end of the adjacency list for node t_a.
     */
    std::vector<Edge*>::iterator const get_edge(int32_t t_a, int32_t t_b)
    {
        return std::find_if(nodes[t_a - 1].begin(), nodes[t_a - 1].end(), [this, t_a, t_b](Edge* edge) {
            return (t_a == edge->get_source() || t_a == edge->get_destination()) && (t_b == edge->get_source() || t_b == edge->get_destination());
        });
    };
};

/**
 * @struct InGraph
 * @brief Structure representing a graph.
 *
 * The Graph consists of nodes which are connected by edges. Each node is
 * represented by an integer identifier and stores a list of edges. Additionally,
 * some nodes are designated as terminal nodes.
 */
struct InGraph {
    AdjacencyList adj_list {}; ///> Adjacency list of the input graph.
    std::unordered_set<int32_t> terminal_nodes {}; ///> List of terminal node IDs.
};

/**
 * @struct OutGraph
 * @brief Output graph. Contains the all mst edges corrsponding to input graph and total weight.
 */
struct OutGraph {
    AdjacencyList adj_list {}; ///> Adjacency list of the input graph.
    std::vector<std::pair<int32_t, int32_t>> result_path {}; ///> Result mst path edges in int pair format.
};

#endif
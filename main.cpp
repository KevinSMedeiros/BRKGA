#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <utility>

struct Edge
{
    int to;
    int weight;
    int capacity;
};

struct Node
{
    int vertex;
    int dist;
    bool operator>(const Node &other) const
    {
        return dist > other.dist;
    }
};

std::vector<double> geraCromossomo(int tamanho)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<double> cromossomo(tamanho);
    for (int i = 0; i < tamanho; ++i)
    {
        do
            cromossomo[i] = dis(gen);
        while (cromossomo[i] == 1.0);
    }
    return cromossomo;
}

std::vector<double> crossOver(const std::vector<double> &elite, const std::vector<double> &nonElite, double bias)
{
    int tamanho = elite.size();
    std::vector<double> filho(tamanho);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < tamanho; ++i)
    {
        if (dis(gen) < bias)
        {
            filho[i] = elite[i];
        }
        else
        {
            filho[i] = nonElite[i];
        }
    }
    return filho;
}

void findECMPRoutes(int start, int target, const std::vector<std::vector<Edge>> &graph, std::vector<std::vector<int>> &parentList)
{
    int n = graph.size();
    std::vector<int> dist(n, std::numeric_limits<int>::max());
    parentList.assign(n, std::vector<int>());
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    dist[start] = 0;
    pq.push({start, 0});

    while (!pq.empty())
    {
        Node current = pq.top();
        pq.pop();
        int u = current.vertex;
        int d = current.dist;

        if (d > dist[u])
            continue;

        if (u == target) break;

        for (const auto &edge : graph[u])
        {
            int v = edge.to;
            int weight = edge.weight;

            if (dist[u] + weight < dist[v])
            {
                dist[v] = dist[u] + weight;
                parentList[v].clear();
                parentList[v].push_back(u);
                pq.push({v, dist[v]});
            }
            else if (dist[u] + weight == dist[v])
            {
                parentList[v].push_back(u);
            }
        }
    }
}

std::vector<int> decoder(const std::vector<double> &cromossomo, const std::vector<std::vector<Edge>> &graph, int start, int target, int maxSeg)
{
    struct PairDoubleCompare
    {
        bool operator()(const std::pair<int, double> &p, double val) const
        {
            return p.second < val;
        }
        bool operator()(double val, const std::pair<int, double> &p) const
        {
            return val < p.second;
        }
    };
    // cromossomo[0] é o gene que indica a quantidade de waypoints
    int quantidadeWaypoints = static_cast<int>(cromossomo[0] * maxSeg);
    std::vector<std::pair<int, double>> vCand;
    std::vector<int> p;

    if (quantidadeWaypoints == 0 || graph.size() <= 2)
    {
        return {start, target};
    }

    for (int i = 1; i < graph.size(); ++i)
    {
        if (i - 1 == start || i - 1 == target)
            continue;
        if (vCand.size() == 0)
        {
            vCand.push_back({i - 1, cromossomo[i]});
            continue;
        }
        auto it = std::upper_bound(vCand.begin(), vCand.end(), cromossomo[i], PairDoubleCompare());
        vCand.insert(it, {i - 1, cromossomo[i]});
    }

    p.push_back(start);
    for (int i = 0; i < quantidadeWaypoints; ++i)
    {
        p.push_back(vCand[i].first);
    }
    p.push_back(target);
    return p;
}

std::vector<double> calculaVetorCarga(const std::vector<int> &p, const std::vector<std::vector<Edge>> &graph)
{
    std::vector<double> vetorCarga;
    
    return vetorCarga;
}

int main()
{
    int tamanho = 7;
    double bias = 0.7;
    std::vector<double> cromossomo = geraCromossomo(tamanho);

    std::cout << "Cromossomo gerado: ";
    for (double gene : cromossomo)
    {
        std::cout << gene << " ";
    }
    std::cout << std::endl;
    const std::vector<std::vector<Edge>> graph = {
        {{1, 1, 1}, {2, 2, 1}},         // 0
        {{0, 1, 1}, {2, 1, 1}, {3, 3, 1}}, // 1
        {{0, 2, 1}, {1, 1, 1}, {3, 1, 1}}, // 2
        {{1, 3, 1}, {2, 1, 1}, {4, 2, 1}}, // 3
        {{3, 2, 1}, {5, 1, 1}},         // 4
        {{4, 1, 1}}                  // 5
    };
    std::vector<int> decoded = decoder(cromossomo, graph, 0, 5, 5);
    std::vector<std::vector<int>> list;
    for (int i = 1; i < decoded.size(); ++i)
    {
       list.clear();
       findECMPRoutes(decoded[i - 1], decoded[i], graph, list);
       std::cout << "\nCaminhos encontrados entre " << decoded[i - 1] << " e " << decoded[i] << ": ";
       for (const auto &parents : list)
         {
              for (int parent : parents)
              {
                std::cout << parent << " ";
              }
              std::cout << "| ";
         }
    }
    std::cout << std::endl;

    return 0;
}
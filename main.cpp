#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <utility>
#include <limits>
#include <numeric>
#include <functional>
#include <fstream>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct Edge
{
    int to;
    int weight;
    int capacity;
    int id;
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

struct Demanda
{
    int source;
    int target;
    double volume;
};

struct Intervencao
{
    std::vector<int> edgeIds;
    int t;
};

struct individuo
{
    std::vector<double> cromossomo;
    std::vector<double> fitness;
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
        filho[i] = (dis(gen) < bias) ? elite[i] : nonElite[i];
    }
    return filho;
}

// ---------------------------------------------------------------------
// Dijkstra com registro de predecessores (necessário para o ECMP)
//
// parentList[v] guarda todos os nós u tais que a aresta u->v participa
// de PELO MENOS UM caminho mínimo de 'start' até v. Com isso conseguimos
// reconstruir, mais adiante, o DAG de caminhos mais curtos usado pelo
// espalhamento ECMP.
// ---------------------------------------------------------------------

void dijkstraComPredecessores(int start, const std::vector<std::vector<Edge>> &graph,
                              std::vector<int> &dist, std::vector<std::vector<int>> &parentList)
{
    int n = graph.size();
    dist.assign(n, std::numeric_limits<int>::max());
    parentList.assign(n, {});
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

        for (const auto &edge : graph[u])
        {
            int v = edge.to;
            int weight = edge.weight;

            if (dist[u] + weight < dist[v])
            {
                dist[v] = dist[u] + weight;
                parentList[v] = {u};
                pq.push({v, dist[v]});
            }
            else if (dist[u] + weight == dist[v])
            {
                parentList[v].push_back(u);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Decoder BRKGA para demanda única estática (Algoritmo 1 do texto)
//
//   Passo 1: l = floor(K_size * (l_max + 1)), l_max = maxSeg - 1
//            como l_max + 1 = maxSeg, temos l = floor(K_size * maxSeg)
//   Passo 2: V_cand = V \ {s,t}, ordenado de forma DECRESCENTE por K_v
//            W = os primeiros l nós de V_cand
//   Passo 3: se l = 0, p = <s,t>; senão p = <s, w1,...,wl, t>
//
// Convenção de cromossomo: gene 0 = K_size; genes 1..|V| = K_v do nó v
// (v indo de 0 a |V|-1), logo o cromossomo tem tamanho N = |V| + 1.
// ---------------------------------------------------------------------

std::vector<int> decoder(const std::vector<double> &cromossomo,
                         const std::vector<std::vector<Edge>> &graph,
                         int start, int target, int maxSeg)
{
    int n = static_cast<int>(graph.size());

    if (n <= 2)
        return {start, target};

    double kSize = cromossomo[0];
    int l = static_cast<int>(kSize * maxSeg); // = floor(K_size * maxSeg), l em [0, maxSeg -]

    if (l == 0)
        return {start, target}; // roteamento direto via caminho mais curto

    // Seleciona candidatos (todos os nós exceto origem e destino) com sua prioridade K_v
    std::vector<std::pair<int, double>> vCand;
    vCand.reserve(n);
    for (int v = 0; v < n; ++v)
    {
        if (v == start || v == target)
            continue;
        double prioridade = cromossomo[v + 1];
        vCand.push_back({v, prioridade});
    }

    // Ordena em ordem DECRESCENTE de prioridade: maiores chaves primeiro
    std::sort(vCand.begin(), vCand.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b)
              { return a.second > b.second; });

    l = std::min<int>(l, static_cast<int>(vCand.size())); // segurança

    std::vector<int> p;
    p.push_back(start);
    for (int i = 0; i < l; ++i)
        p.push_back(vCand[i].first);
    p.push_back(target);
    return p;
}

// ---------------------------------------------------------------------
// Cálculo de fitness
//
// Para uma demanda (s,t) roteada através do caminho em segmentos p, o
// tráfego de cada segmento (i,j) consecutivo em p é espalhado sobre os
// caminhos mais curtos IGP entre i e j segundo as regras do ECMP:
// em cada nó u do DAG de caminhos mínimos, a fração de tráfego que
// chega a u é dividida IGUALMENTE entre todos os seus sucessores no DAG.
//
// r(i,j,a) é a fração do tráfego do segmento (i,j) que atravessa o
// arco a, obtida por essa divisão recursiva. A carga de cada arco é:
//
//     lambda(a) = ( sum_{(i,j) em p} r(i,j,a) * nu ) / c(a)
//
// O fitness do indivíduo é o vetor L = { lambda(a) : a em A }, ordenado
// de forma DECRESCENTE, comparado lexicograficamente no torneio.
// ---------------------------------------------------------------------

// Constrói o grafo reverso (usado para achar, a partir de j, a distância
// de CADA nó até j — necessário para restringir o DAG de ECMP apenas às
// arestas que realmente pertencem a algum caminho mínimo de i até j).
std::vector<std::vector<Edge>> construirGrafoReverso(const std::vector<std::vector<Edge>> &graph)
{
    int n = static_cast<int>(graph.size());
    std::vector<std::vector<Edge>> reverso(n);
    for (int u = 0; u < n; ++u)
        for (const auto &e : graph[u])
            reverso[e.to].push_back({u, e.weight, e.capacity, e.id});
    return reverso;
}

// Acumula, para um único segmento (i,j) do caminho, a parcela de tráfego
// nu * r(i,j,a) em cada arco 'a', somando dentro de cargaAcumulada.
// cargaAcumulada[u][e] refere-se à e-ésima aresta que sai do nó u.
//
// IMPORTANTE: o DAG de espalhamento ECMP não pode ser "toda a árvore de
// Dijkstra a partir de i" — um nó u pode ter, na árvore de Dijkstra
// completa, sucessores que levam a destinos totalmente alheios a j (por
// exemplo, se dois waypoints diferentes do caminho compartilham um nó
// intermediário C, mas seguem para arcos distintos depois de C). Se a
// gente dividir o fluxo entre TODOS os sucessores de Dijkstra, parte do
// tráfego vaza para arcos que não têm nada a ver com o segmento (i,j).
//
// A forma correta é restringir o DAG apenas às arestas (u,v) que
// pertencem a ALGUM caminho mínimo de i até j especificamente, o que se
// testa com: dist_i(u) + peso(u,v) + dist_paraJ(v) == dist_i(j), onde
// dist_paraJ é calculado com um Dijkstra a partir de j no grafo reverso.
void acumulaCargaSegmento(int i, int j, double volume,
                          const std::vector<std::vector<Edge>> &graph,
                          const std::vector<std::vector<Edge>> &grafoReverso,
                          std::vector<std::vector<double>> &cargaAcumulada)
{
    int n = static_cast<int>(graph.size());

    std::vector<int> distDeI, distParaJ;
    std::vector<std::vector<int>> predIgnorado;
    dijkstraComPredecessores(i, graph, distDeI, predIgnorado);
    dijkstraComPredecessores(j, grafoReverso, distParaJ, predIgnorado); // dist(j->v) no reverso = dist(v->j) no original

    if (distDeI[j] == std::numeric_limits<int>::max())
        return; // não há caminho entre i e j; nada a acumular

    int total = distDeI[j];

    // sucessoresValidos[u] = lista de (v, índice da aresta em graph[u])
    // que pertencem a algum caminho mínimo de i até j
    std::vector<std::vector<std::pair<int, int>>> sucessoresValidos(n);
    for (int u = 0; u < n; ++u)
    {
        if (distDeI[u] == std::numeric_limits<int>::max())
            continue;
        for (size_t e = 0; e < graph[u].size(); ++e)
        {
            int v = graph[u][e].to;
            int w = graph[u][e].weight;
            if (distParaJ[v] == std::numeric_limits<int>::max())
                continue;
            if (distDeI[u] + w + distParaJ[v] == total)
                sucessoresValidos[u].push_back({v, static_cast<int>(e)});
        }
    }

    // Processa os nós em ordem crescente de distância (ordem topológica do DAG restrito)
    std::vector<int> ordem(n);
    std::iota(ordem.begin(), ordem.end(), 0);
    std::sort(ordem.begin(), ordem.end(),
              [&](int a, int b)
              { return distDeI[a] < distDeI[b]; });

    // fluxo[v] = fração do tráfego do segmento (i,j) que chega ao nó v
    std::vector<double> fluxo(n, 0.0);
    fluxo[i] = 1.0;

    for (int u : ordem)
    {
        if (u == j || fluxo[u] <= 0.0 || sucessoresValidos[u].empty())
            continue; // não propaga além do destino do segmento

        double parcela = fluxo[u] / static_cast<double>(sucessoresValidos[u].size()); // divisão ECMP

        for (const auto &par : sucessoresValidos[u])
        {
            int v = par.first;
            int e = par.second;
            fluxo[v] += parcela;
            cargaAcumulada[u][e] += parcela * volume; // r(i,j,a) * volume
        }
    }
}

// Monta o vetor de fitness L = { lambda(a) : a em A } para o caminho p
// completo (todos os segmentos), já ordenado de forma decrescente.
std::vector<double> calculaVetorCarga(const std::vector<int> &p, double volume,
                                      const std::vector<std::vector<Edge>> &graph, const std::vector<std::vector<Edge>> &grafoReverso)
{
    int n = static_cast<int>(graph.size());

    std::vector<std::vector<double>> cargaAcumulada(n);
    for (int u = 0; u < n; ++u)
        cargaAcumulada[u].assign(graph[u].size(), 0.0);

    for (size_t k = 1; k < p.size(); ++k)
        acumulaCargaSegmento(p[k - 1], p[k], volume, graph, grafoReverso, cargaAcumulada);

    std::vector<double> L;
    for (int u = 0; u < n; ++u)
        for (size_t e = 0; e < graph[u].size(); ++e)
            L.push_back(cargaAcumulada[u][e] / graph[u][e].capacity);

    std::sort(L.begin(), L.end(), std::greater<double>());
    return L;
}

// Comparação lexicográfica de dois vetores de fitness: retorna true se
// 'a' é MELHOR que 'b' (minimiza primeiro a maior carga, depois a
// segunda maior, e assim sucessivamente — usado no torneio do BRKGA).
bool melhorFitness(const std::vector<double> &a, const std::vector<double> &b)
{
    size_t n = std::min(a.size(), b.size());
    for (size_t idx = 0; idx < n; ++idx)
        if (a[idx] != b[idx])
            return a[idx] < b[idx];
    return a.size() < b.size();
}

// Função de fitness completa: decodifica o cromossomo, roteia a demanda
// e devolve o vetor de carga L já ordenado (usado na comparação
// lexicográfica). O valor de retorno (a maior carga) é apenas um resumo
// útil para logs/relatórios.
double fitness(const std::vector<double> &cromossomo,
               const std::vector<std::vector<Edge>> &graph,
               const std::vector<std::vector<Edge>> &grafoReverso,
               int start, int target, int maxSeg, double volume,
               std::vector<double> &vetorCarga)
{
    std::vector<int> p = decoder(cromossomo, graph, start, target, maxSeg);
    vetorCarga = calculaVetorCarga(p, volume, graph, grafoReverso);
    return vetorCarga.empty() ? 0.0 : vetorCarga.front();
}

void carregaGrafoDeArquivo(const std::string &nomeArquivo, std::vector<std::vector<Edge>> &graph)
{
    std::ifstream arquivo(nomeArquivo);
    if (!arquivo.is_open())
    {
        std::cerr << "Erro ao abrir o arquivo: " << nomeArquivo << std::endl;
        return;
    }

    json j;
    arquivo >> j;

    int numVertices = j["nodes"].size();
    graph.resize(numVertices);

    for (const auto &aresta : j["links"])
    {
        int from = aresta["from"];
        int id = aresta["id"].get<int>();
        int to = aresta["to"];
        int weight = aresta["metric"];
        int capacity = aresta["capacity"];
        graph[from].push_back({to, weight, capacity, id});
    }
}
void carregaMaxSegEIntervencoesDeArquivo(const std::string &nomeArquivo, int &maxSeg, std::vector<Intervencao> &intervencoes)
{
    std::ifstream arquivo(nomeArquivo);
    if (!arquivo.is_open())
    {
        std::cerr << "Erro ao abrir o arquivo: " << nomeArquivo << std::endl;
        return;
    }

    json j;
    arquivo >> j;
    std::vector<int> edgeIds;

    maxSeg = j["max_segments"].get<int>();
    for (const auto &intervencao : j["interventions"])
    {
        int t = intervencao["t"];
        edgeIds.clear();
        for (const auto &edgeId : intervencao["links"])
        {
            edgeIds.push_back(edgeId);
        }
        intervencoes.push_back({edgeIds, t});
    }
}

void carregaDemandasDeArquivo(const std::string &nomeArquivo, std::vector<Demanda> &demandas)
{
    std::ifstream arquivo(nomeArquivo);
    if (!arquivo.is_open())
    {
        std::cerr << "Erro ao abrir o arquivo: " << nomeArquivo << std::endl;
        return;
    }

    json j;
    arquivo >> j;
    int numVolumes = j["num_time_slots"].get<int>();

    for (const auto &demanda : j["demands"])
    {
        for (int i = 0; i < numVolumes; ++i)
        {
            int source = demanda["s"];
            int target = demanda["t"];
            double volume = demanda["v"][i].get<double>();
            demandas.push_back({source, target, volume});
        }
    }
}

void executaIntervencoes(const std::vector<Intervencao> &intervencoes, std::vector<std::vector<Edge>> &graph, int tempoAtual)
{
}


void BRKGA(const std::vector<std::vector<Edge>> &graph, int maxSeg, int tamanhoPopulacao, int tamanhoElite, int tamanhoMutante, int numGeracoes, double bias, individuo &melhorIndividuo, const Demanda &demanda)
{
    int tamanhoCromossomo = static_cast<int>(graph.size()) + 1; // N = |V| + 1
    std::vector<individuo> populacao(tamanhoPopulacao);
    const std::vector<std::vector<Edge>> grafoReverso = construirGrafoReverso(graph);
    for (int i = 0; i < tamanhoPopulacao; ++i)
    {
        populacao[i].cromossomo = geraCromossomo(tamanhoCromossomo);
        fitness(populacao[i].cromossomo, graph, grafoReverso, demanda.source, demanda.target, maxSeg, demanda.volume, populacao[i].fitness);
    }

    std::sort(populacao.begin(), populacao.end(), [](const individuo &a, const individuo &b)
              { return melhorFitness(a.fitness, b.fitness); });

    melhorIndividuo = populacao[0];
    std::cout << "Fitness inicial: " << melhorIndividuo.fitness[0] << std::endl;

    for (int geracao = 0; geracao < numGeracoes; ++geracao)
    {
        std::vector<individuo> novaPopulacao;

        for (int i = 0; i < tamanhoElite; ++i)
            novaPopulacao.push_back(populacao[i]);

        while (static_cast<int>(novaPopulacao.size()) < tamanhoPopulacao - tamanhoMutante)
        {
            int idxElite = rand() % tamanhoElite;
            int idxNonElite = tamanhoElite + rand() % (tamanhoPopulacao - tamanhoElite);
            std::vector<double> filhoCromossomo = crossOver(populacao[idxElite].cromossomo, populacao[idxNonElite].cromossomo, bias);

            individuo filho;
            filho.cromossomo = filhoCromossomo;
            fitness(filho.cromossomo, graph, grafoReverso, demanda.source, demanda.target, maxSeg, demanda.volume, filho.fitness);
            novaPopulacao.push_back(filho);
        }

        for (int i = 0; i < tamanhoMutante; ++i)
        {
            std::vector<double> mutanteCromossomo = geraCromossomo(tamanhoCromossomo);
            individuo mutante;
            mutante.cromossomo = mutanteCromossomo;
            fitness(mutante.cromossomo, graph, grafoReverso, demanda.source, demanda.target, maxSeg, demanda.volume, mutante.fitness);
            novaPopulacao.push_back(mutante);
        }

        populacao = novaPopulacao;

        std::sort(populacao.begin(), populacao.end(), [](const individuo &a, const individuo &b)
                  { return melhorFitness(a.fitness, b.fitness); });
        std::cout << "Geração " << geracao + 1 << ": Melhor fitness = " << std::endl;
        for (const auto &f : populacao[0].fitness)
            std::cout << f << " ";
            std::cout << std::endl;
    }
}
int main()
{
    std::vector<Demanda> demandas;
    int maxSeg;
    std::vector<std::vector<Edge>> graph;
    std::vector<Intervencao> intervencoes;
    individuo melhorIndividuo;

    carregaDemandasDeArquivo("instances/setA/setA-02-tm.json", demandas);
    carregaMaxSegEIntervencoesDeArquivo("instances/setA/setA-02-scenario.json", maxSeg, intervencoes);
    carregaGrafoDeArquivo("instances/setA/setA-02-net.json", graph);

    if (demandas.empty())
    {
        std::cerr << "Nenhuma demanda carregada." << std::endl;
        return 1;
    }



    BRKGA(graph, maxSeg, 10, 4, 2, 10, 0.7, melhorIndividuo, demandas[10]);
    return 0;
}
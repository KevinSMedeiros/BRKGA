#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    double num = 0;
    bool b = false;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    const Json& at(const std::string& key) const {
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("chave ausente: " + key);
        return it->second;
    }

    bool has(const std::string& key) const { return obj.count(key) > 0; }
};

class Parser {
    const std::string& text;
    size_t i = 0;

    void ws() {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    }

    [[noreturn]] void fail(const std::string& message) {
        throw std::runtime_error("JSON invalido na posicao " + std::to_string(i) + ": " + message);
    }

    char peek() {
        ws();
        if (i >= text.size()) fail("fim inesperado");
        return text[i];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("esperado '") + c + "'");
        ++i;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (i < text.size() && text[i] != '"') {
            if (text[i] == '\\' && i + 1 < text.size()) {
                ++i;
                switch (text[i]) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': out += '?'; i += 4; break;
                    default: out += text[i];
                }
            } else {
                out += text[i];
            }
            ++i;
        }
        if (i >= text.size()) fail("string nao terminada");
        ++i;
        return out;
    }

public:
    explicit Parser(const std::string& input) : text(input) {}

    Json parse() {
        Json value;
        const char c = peek();
        if (c == '{') {
            value.type = Json::Obj;
            ++i;
            if (peek() == '}') {
                ++i;
                return value;
            }
            while (true) {
                ws();
                std::string key = parseString();
                expect(':');
                value.obj[key] = parse();
                if (peek() == ',') {
                    ++i;
                    continue;
                }
                expect('}');
                break;
            }
        } else if (c == '[') {
            value.type = Json::Arr;
            ++i;
            if (peek() == ']') {
                ++i;
                return value;
            }
            while (true) {
                value.arr.push_back(parse());
                if (peek() == ',') {
                    ++i;
                    continue;
                }
                expect(']');
                break;
            }
        } else if (c == '"') {
            value.type = Json::Str;
            value.str = parseString();
        } else if (text.compare(i, 4, "true") == 0) {
            value.type = Json::Bool;
            value.b = true;
            i += 4;
        } else if (text.compare(i, 5, "false") == 0) {
            value.type = Json::Bool;
            value.b = false;
            i += 5;
        } else if (text.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            const size_t start = i;
            while (i < text.size() &&
                   (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '-' ||
                    text[i] == '+' || text[i] == '.' || text[i] == 'e' || text[i] == 'E')) {
                ++i;
            }
            if (start == i) fail("token inesperado");
            value.type = Json::Num;
            value.num = std::stod(text.substr(start, i - start));
        }
        return value;
    }
};

struct Registro {
    int source = 0;
    int target = 0;
    std::string numWayPoints = "-";
    double volume = 0;
    std::vector<int> rota;
    std::vector<double> fitness;
    double tempoMs = 0;

    double melhorFitness() const {
        return fitness.empty() ? 0.0 : *std::max_element(fitness.begin(), fitness.end());
    }

    int solucoesValidas() const {
        return static_cast<int>(std::count_if(fitness.begin(), fitness.end(),
                                              [](double f) { return f > 0; }));
    }

    std::string rotaStr() const {
        std::string route;
        for (size_t k = 0; k < rota.size(); ++k) {
            route += (k ? ">" : "") + std::to_string(rota[k]);
        }
        return route;
    }
};

struct Arquivo {
    std::string nome;
    std::string instancia;
    std::vector<Registro> regs;
    std::string params;
};

struct ResultadoComparacao {
    std::vector<std::vector<std::string>> linhas;
    double somaTA = 0;
    double somaTB = 0;
    double somaFA = 0;
    double somaFB = 0;
    int vitA = 0;
    int vitB = 0;
    int empates = 0;
};

static const std::vector<std::string> CABECALHO = {
    "Instancia", "Demanda", "Volume", "NumWayPoints A", "NumWayPoints B",
    "Rota A", "Rota B", "Tempo A(ms)", "Tempo B(ms)", "Speedup A/B",
    "Umax A", "Umax B", "Var. Umax", "Arcos>0 A", "Arcos>0 B",
    "1a dif.", "Ganho", "Melhor"};

static std::string lerArquivo(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("nao foi possivel abrir: " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static Arquivo carregar(const std::string& path) {
    const Json root = Parser(lerArquivo(path)).parse();
    if (root.type != Json::Arr) {
        throw std::runtime_error("raiz de " + path + " deve ser um array");
    }

    Arquivo arquivo;
    const size_t slash = path.find_last_of("/\\");
    arquivo.nome = path.substr(slash == std::string::npos ? 0 : slash + 1);
    size_t registroIndex = 0;
    for (const Json& item : root.arr) {
        Registro registro;
        const Json& demanda = item.at("demanda");
        registro.source = static_cast<int>(demanda.at("source").num);
        registro.target = static_cast<int>(demanda.at("target").num);
        registro.volume = demanda.at("volume").num;
        if (item.has("numWayPoints")) {
            const Json& numWayPoints = item.at("numWayPoints");
            if (numWayPoints.type != Json::Num) {
                throw std::runtime_error(path + ": numWayPoints nao numerico no registro " +
                                         std::to_string(registroIndex));
            }
            registro.numWayPoints = std::to_string(static_cast<int>(numWayPoints.num));
        }
        for (const Json& node : item.at("esquemaRoteamento").arr) {
            registro.rota.push_back(static_cast<int>(node.num));
        }

        const Json& fitness = item.at("fitness");
        if (fitness.type != Json::Arr) {
            throw std::runtime_error(path + ": fitness do registro " +
                                     std::to_string(registroIndex) + " deve ser um array");
        }
        for (size_t fitnessIndex = 0; fitnessIndex < fitness.arr.size(); ++fitnessIndex) {
            const Json& value = fitness.arr[fitnessIndex];
            if (value.type == Json::Null) {
                throw std::runtime_error(path + ": fitness null no registro " +
                                         std::to_string(registroIndex) + ", indice " +
                                         std::to_string(fitnessIndex));
            }
            if (value.type != Json::Num) {
                throw std::runtime_error(path + ": fitness nao numerico no registro " +
                                         std::to_string(registroIndex) + ", indice " +
                                         std::to_string(fitnessIndex));
            }
            registro.fitness.push_back(value.num == 0.0 ? 0.0 : value.num);
        }

        registro.tempoMs = item.at("tempoExecucao").num;
        if (arquivo.instancia.empty() && item.has("instancia")) {
            arquivo.instancia = item.at("instancia").str;
        }
        if (arquivo.params.empty() && item.has("parametros")) {
            const Json& parametros = item.at("parametros");
            std::ostringstream stream;
            stream << "geracoes=" << static_cast<long>(parametros.at("numGeracoes").num)
                   << " populacao=" << static_cast<long>(parametros.at("tamanhoPopulacao").num)
                   << " elite=" << static_cast<long>(parametros.at("tamanhoElite").num)
                   << " mutantes=" << static_cast<long>(parametros.at("tamanhoMutante").num)
                   << " bias=" << parametros.at("bias").num;
            arquivo.params = stream.str();
        }
        arquivo.regs.push_back(std::move(registro));
        ++registroIndex;
    }
    return arquivo;
}

static bool melhorFitness(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return a.size() < b.size();
}

static int primeiraDiferenca(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return static_cast<int>(i);
    }
    return a.size() != b.size() ? static_cast<int>(n) : -1;
}

static std::string fmt(double value, int precision = 4) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

static std::string pct(double value) {
    std::ostringstream stream;
    stream << std::showpos << std::fixed << std::setprecision(2) << value << "%";
    return stream.str();
}

static std::pair<int, int> chaveDemanda(const Registro& registro) {
    return {registro.source, registro.target};
}

static ResultadoComparacao comparar(const Arquivo& a, const Arquivo& b, bool estrito) {
    std::map<std::pair<int, int>, const Registro*> registrosB;
    for (const Registro& registro : b.regs) {
        if (!registrosB.emplace(chaveDemanda(registro), &registro).second) {
            throw std::runtime_error("demanda duplicada em " + b.nome + ": " +
                                     std::to_string(registro.source) + "->" +
                                     std::to_string(registro.target));
        }
    }

    ResultadoComparacao resultado;
    std::set<std::pair<int, int>> demandasComparadas;
    const std::string instancia = a.instancia.empty() ? a.nome : a.instancia;
    for (const Registro& ra : a.regs) {
        const auto key = chaveDemanda(ra);
        if (!demandasComparadas.insert(key).second) {
            throw std::runtime_error("demanda duplicada em " + a.nome + ": " +
                                     std::to_string(ra.source) + "->" +
                                     std::to_string(ra.target));
        }
        const auto it = registrosB.find(key);
        if (it == registrosB.end()) {
            if (estrito) {
                throw std::runtime_error("demanda " + std::to_string(ra.source) + "->" +
                                         std::to_string(ra.target) + " ausente em " + b.nome);
            }
            std::cerr << "Aviso: demanda " << ra.source << "->" << ra.target
                      << " ausente em B\n";
            continue;
        }

        const Registro& rb = *it->second;
        const double fa = ra.melhorFitness();
        const double fb = rb.melhorFitness();
        const double variacao = fa > 0 ? (fb - fa) / fa * 100.0 : (fb > 0 ? 100.0 : 0.0);
        const double speedup = rb.tempoMs > 0 ? ra.tempoMs / rb.tempoMs : 0.0;
        std::string melhor;
        if (melhorFitness(ra.fitness, rb.fitness)) {
            melhor = "A";
            ++resultado.vitA;
        } else if (melhorFitness(rb.fitness, ra.fitness)) {
            melhor = "B";
            ++resultado.vitB;
        } else {
            melhor = "Empate";
            ++resultado.empates;
        }

        const int diferenca = primeiraDiferenca(ra.fitness, rb.fitness);
        std::string ganho = "-";
        if (diferenca >= 0 &&
            diferenca < static_cast<int>(std::min(ra.fitness.size(), rb.fitness.size()))) {
            const double x = ra.fitness[diferenca];
            const double y = rb.fitness[diferenca];
            const double maior = std::max(x, y);
            const double menor = std::min(x, y);
            ganho = fmt(maior > 0 ? (maior - menor) / maior * 100.0 : 0.0, 2) +
                    "% (" + melhor + ")";
        }

        resultado.linhas.push_back({
            instancia,
            std::to_string(ra.source) + "->" + std::to_string(ra.target),
            fmt(ra.volume),
            ra.numWayPoints,
            rb.numWayPoints,
            ra.rotaStr(),
            rb.rotaStr(),
            fmt(ra.tempoMs, 0),
            fmt(rb.tempoMs, 0),
            fmt(speedup, 1) + "x",
            fmt(fa, 6),
            fmt(fb, 6),
            pct(variacao),
            std::to_string(ra.solucoesValidas()),
            std::to_string(rb.solucoesValidas()),
            diferenca < 0 ? "-" : std::to_string(diferenca),
            ganho,
            melhor});
        resultado.somaTA += ra.tempoMs;
        resultado.somaTB += rb.tempoMs;
        resultado.somaFA += fa;
        resultado.somaFB += fb;
    }

    if (estrito && demandasComparadas.size() != registrosB.size()) {
        for (const auto& entry : registrosB) {
            if (demandasComparadas.count(entry.first) == 0) {
                throw std::runtime_error("demanda " + std::to_string(entry.first.first) + "->" +
                                         std::to_string(entry.first.second) + " ausente em " +
                                         a.nome);
            }
        }
    }
    if (resultado.linhas.empty()) {
        throw std::runtime_error("nenhuma demanda em comum entre " + a.nome + " e " + b.nome);
    }
    return resultado;
}

static std::string escaparCsv(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

static void escreverCsv(const fs::path& path, const std::vector<std::vector<std::string>>& rows) {
    std::ofstream csv(path);
    if (!csv) throw std::runtime_error("nao foi possivel criar o CSV: " + path.string());
    for (size_t i = 0; i < CABECALHO.size(); ++i) {
        csv << (i ? "," : "") << escaparCsv(CABECALHO[i]);
    }
    csv << '\n';
    for (const auto& row : rows) {
        if (row.size() != CABECALHO.size()) {
            throw std::runtime_error("quantidade de colunas invalida ao escrever " + path.string());
        }
        for (size_t i = 0; i < row.size(); ++i) {
            csv << (i ? "," : "") << escaparCsv(row[i]);
        }
        csv << '\n';
    }
    if (!csv) throw std::runtime_error("erro ao escrever o CSV: " + path.string());
}

static std::string normalizar(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized;
}

static bool arquivoResultado(const std::string& name) {
    const std::string normalized = normalizar(name);
    return normalized.size() == 19 && normalized.compare(0, 3, "set") == 0 &&
           (normalized[3] == 'a' || normalized[3] == 'b' ||
            normalized[3] == 'c' || normalized[3] == 'x') &&
           normalized[4] == '-' && std::isdigit(static_cast<unsigned char>(normalized[5])) &&
           std::isdigit(static_cast<unsigned char>(normalized[6])) &&
           normalized.substr(7) == "-result.json";
}

static std::string nomeSet(const std::string& filename) {
    std::string set = normalizar(filename.substr(0, 4));
    set[3] = static_cast<char>(std::toupper(static_cast<unsigned char>(set[3])));
    return set;
}

static std::map<std::string, fs::path> listarResultados(const fs::path& directory) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("pasta de resultados inexistente: " + directory.string());
    }
    std::map<std::string, fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!arquivoResultado(name)) continue;
        const std::string key = normalizar(name);
        if (!files.emplace(key, entry.path()).second) {
            throw std::runtime_error("arquivo duplicado ignorando maiusculas/minusculas: " + name);
        }
    }
    if (files.empty()) {
        throw std::runtime_error("nenhum arquivo de resultado encontrado em " + directory.string());
    }
    return files;
}

void compararPastas(const std::string& pastaA, const std::string& pastaB,
                    const std::string& pastaSaida) {
    const auto arquivosA = listarResultados(pastaA);
    const auto arquivosB = listarResultados(pastaB);
    if (arquivosA.size() != arquivosB.size()) {
        throw std::runtime_error("as pastas possuem quantidades diferentes de arquivos de resultado");
    }

    std::map<std::string, std::vector<std::vector<std::string>>> linhasPorSet;
    for (const auto& entry : arquivosA) {
        const auto itB = arquivosB.find(entry.first);
        if (itB == arquivosB.end()) {
            throw std::runtime_error("arquivo correspondente ausente em " + pastaB + ": " +
                                     entry.second.filename().string());
        }
        const Arquivo a = carregar(entry.second.string());
        const Arquivo b = carregar(itB->second.string());
        ResultadoComparacao comparison = comparar(a, b, true);
        std::vector<std::vector<std::string>>& setRows = linhasPorSet[nomeSet(entry.first)];
        setRows.insert(setRows.end(), comparison.linhas.begin(), comparison.linhas.end());
    }
    for (const auto& entry : arquivosB) {
        if (arquivosA.count(entry.first) == 0) {
            throw std::runtime_error("arquivo correspondente ausente em " + pastaA + ": " +
                                     entry.second.filename().string());
        }
    }

    const fs::path outputDirectory(pastaSaida);
    fs::create_directories(outputDirectory);
    for (const auto& entry : linhasPorSet) {
        const fs::path outputPath = outputDirectory / (entry.first + ".csv");
        escreverCsv(outputPath, entry.second);
        std::cout << "CSV salvo em: " << outputPath.string() << " (" << entry.second.size()
                  << " demandas)\n";
    }
}

static void imprimirTabela(const ResultadoComparacao& resultado) {
    std::vector<int> widths;
    for (const std::string& column : CABECALHO) widths.push_back(static_cast<int>(column.size()));
    for (const auto& row : resultado.linhas) {
        for (size_t i = 0; i < row.size(); ++i) {
            widths[i] = std::max(widths[i], static_cast<int>(row[i].size()));
        }
    }
    auto separator = [&widths](char c) {
        std::cout << '+';
        for (int width : widths) std::cout << std::string(width + 2, c) << '+';
        std::cout << '\n';
    };
    auto printRow = [&widths](const std::vector<std::string>& row) {
        std::cout << '|';
        for (size_t i = 0; i < widths.size(); ++i) {
            std::cout << ' ' << std::left << std::setw(widths[i]) << row[i] << " |";
        }
        std::cout << '\n';
    };

    separator('-');
    printRow(CABECALHO);
    separator('=');
    for (const auto& row : resultado.linhas) printRow(row);
    separator('-');
}

static bool ehPasta(const char* path) {
    return fs::is_directory(fs::path(path));
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            compararPastas("resultados", "resultados-greedy", "comparacoes");
            return 0;
        }
        if (argc < 3 || argc > 4) {
            std::cerr << "Uso:\n  " << argv[0]
                      << " [pastaA pastaB [pastaSaida]]\n  " << argv[0]
                      << " arquivoA.json arquivoB.json [saida.csv]\n";
            return 1;
        }

        if (ehPasta(argv[1]) && ehPasta(argv[2])) {
            compararPastas(argv[1], argv[2], argc == 4 ? argv[3] : "comparacoes");
            return 0;
        }

        const Arquivo a = carregar(argv[1]);
        const Arquivo b = carregar(argv[2]);
        const ResultadoComparacao comparison = comparar(a, b, false);
        std::cout << "\n=== QUADRO COMPARATIVO ===\n";
        std::cout << "A: " << a.nome << " [" << a.params << "]\n";
        std::cout << "B: " << b.nome << " [" << b.params << "]\n\n";
        imprimirTabela(comparison);
        const int n = static_cast<int>(comparison.linhas.size());
        const double variacaoMedia = comparison.somaFA > 0
                                         ? (comparison.somaFB - comparison.somaFA) /
                                               comparison.somaFA * 100.0
                                         : 0.0;
        std::cout << "\n--- RESUMO (" << n << " demandas) ---\n";
        std::cout << "Tempo total   A: " << fmt(comparison.somaTA, 0) << " ms | B: "
                  << fmt(comparison.somaTB, 0) << " ms | speedup global A/B: "
                  << fmt(comparison.somaTB > 0 ? comparison.somaTA / comparison.somaTB : 0, 1)
                  << "x\n";
        std::cout << "Tempo medio   A: " << fmt(comparison.somaTA / n, 1) << " ms | B: "
                  << fmt(comparison.somaTB / n, 1) << " ms\n";
        std::cout << "Umax media    A: " << fmt(comparison.somaFA / n, 6) << " | B: "
                  << fmt(comparison.somaFB / n, 6) << " | variacao B vs A: "
                  << pct(variacaoMedia) << " (positivo = B pior)\n";
        std::cout << "Melhor vetor por demanda -> A: " << comparison.vitA
                  << " | B: " << comparison.vitB << " | empates: " << comparison.empates
                  << "\n\n";

        if (argc == 4) {
            escreverCsv(argv[3], comparison.linhas);
            std::cout << "CSV salvo em: " << argv[3] << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Erro: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

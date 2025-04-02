# Algoritmo 1 – Redução Paralela

Foram implementados algoritmos de redução paralela para o cálculo de máximo, mínimo e soma (esta última também utilizada para o cálculo da média) de vetores unidimensionais.

A versão mais simples, implementada na CPU, é totalmente linear, exigindo o acesso a cada índice do vetor de forma sequencial e escalando em $` O(N) `$. Já a versão implementada na GPU adota uma estratégia binária, reduzindo o tamanho do vetor pela metade a cada iteração, resultando em uma complexidade de $` O(\log_2(N)) `$.  

Na prática, devido às limitações de memória da GPU, cada chamada da função pode executar no máximo $` \log_2(\text{Block\_Size}) `$ iterações, exigindo chamadas adicionais caso o vetor seja maior que o tamanho do bloco. Como o acesso à GPU pode ser custoso, a implementação interrompe a recursão caso o tamanho do vetor fique menor que um bloco, finalizando a redução na CPU, que é mais eficiente para vetores pequenos.

Abaixo, apresentamos os tempos de execução aproximados das três funções para diferentes tamanhos de vetores gerados com números aleatórios. Observamos que o tempo de execução na CPU segue próximo a $` O(N) `$, com exceção de $` N = 100 `$, que é mais suscetível a flutuações. No caso da GPU, notamos que há um custo fixo associado à alocação de memória na GPU (aproximadamente 300ms) que domina o termo $` O(\log_2(N)) `$, tornando-a mais lenta para $` N < 2^{20} `$.

| Implementação | CPU (Linear) | GPU (Binary Tree) |
|--------------|-------------|------------------|
| N = 100     | 0.0038ms    | 300ms            |
| N = 1000    | 0.066ms     | 300ms            |
| N = 2^15    | 2.15ms      | 290ms            |
| N = 2^20    | 70ms        | 308ms            |
| N = 2^25    | 2200ms      | 570ms            |

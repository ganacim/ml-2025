Não usei a estrutura do CMake nesse exercício.

Para rodar usei o comando:
>> nvcc -o nome_output project1/src/arquivo_desejado.cu
para criar um executável.


## CUM_SUM

No primeiro caso (soma cumulativa) ainda estava tentando entender melhor a linguagem e usei bastante o chatGPT.
Isso levou a eu ficar em dúvida em algumas partes do código:
- O uso do __syncthreads e o que ocorreria de ruim sem ele.
- O motivo de ser melhor voltar pra cpu na linha 63.

Os comentários em portugues são meus, seja por ser uma parte que eu fiz, seja por ser uma parte que eu anotei para ler depois.



## LIFE

No segundo programa (game of life) tentei fazer um pouco mais independentemente.
Usei principalmente forums como fonte, e IAs mais para tirar dúvidas se como se escreve algo.

Também fiz uma animação de terminal para melhor visualizar o jogo. 



## Executar

Para executar, basta botar no terminal:

./01_cuda/marcelo_carneiro/src/cum_sum

ou 

./01_cuda/marcelo_carneiro/src/life


Por algum motivo funciona melhor sem ser no terminal integrado do vscode.
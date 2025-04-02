# Algoritmo 2 – Conjunto de Mandelbrot

Para melhor visualização do resultado, foi implementado uma interface gráfica com a biblioteca GLUT. A interface permite translado com WASD, zoom com Q e E. Apertar R volta ao padrão. 
Para obter a biblioteca: sudo apt-get install freeglut3-dev mesa-utils.

Determinar se um pixel divergirá ou não pode exigir até N Máximo de iterações. A implementação na CPU calcula linearmente cada pixel da imagem, resultando em uma complexidade de $` O(\text{Max Iter} \times \text{WIDTH} \times \text{HEIGHT}) `$. A implementação na GPU aloca uma thread para cada pixel e calcula todas as iterações paralelamente, sendo então de ordem $` O(\text{Max Iter}) `$. O resultado para um pixel independe de seus vizinhos, o que simplifica a implementação. Não é necessário o uso de _shared_ _memory_ para comunicação entre as threads ou realizar sincronização de _warps_, e a performance é, em maior parte, invariante ao tamanho de bloco.

Abaixo, temos os tempos aproximados para o cálculo do número de iterações até a divergência para cada pixel na posição inicial da imagem. 

| Implementação                 | CPU (Linear)       | GPU (Processamento Paralelo) |
|--------------------------------|-------------------|-----------------------------|
| Max Iter = 20, W = H = 400     | 0.064s / 15.6 FPS | 0.0051s / 195 FPS           |
| Max Iter = 200, W = H = 400    | 0.28s / 3.55 FPS  | 0.0092s / 110 FPS           |
| Max Iter = 2000, W = H = 400   | 2.42s / 0.41 FPS  | 0.057s / 17.5 FPS           |
| Max Iter = 20, W = H = 800     | 0.26s / 3.85 FPS  | 0.045s / 22.2 FPS           |
| Max Iter = 200, W = H = 800    | 1.14s / 0.88 FPS  | 0.088s / 11.3 FPS           |
| Max Iter = 2000, W = H = 800   | 9.6s / 0.10 FPS   | 0.17s / 5.73 FPS            |
